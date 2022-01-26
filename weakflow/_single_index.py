import copy  # For deepcopy function
import math
import time

import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, check_random_state, check_X_y

from ddl.base import (CompositeDestructor, DestructorMixin,
                      create_inverse_transformer)
from ddl.independent import (IndependentDensity, IndependentDestructor,
                             IndependentInverseCdf)


__all__ = ['SingleIndexEnsembleEstimator',
           'SingleIndexEnsembleModel']


class SingleIndexEnsembleEstimator(BaseEstimator, ClassifierMixin):
    """Scikit-learn estimator for PyTorch model saves classes_ and model_"""
    def __init__(self, model=None, get_optimizer=None,
                 batch_size=500, train_steps=1000, shuffle=True,
                 use_cuda=False, cross_entropy_eps=1e-7,
                 train_callback=None, verbosity=1,
                 random_state=None):
        self.model = model
        self.get_optimizer = get_optimizer
        self.cross_entropy_eps = cross_entropy_eps
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.shuffle = shuffle
        self.use_cuda = use_cuda
        self.train_callback = train_callback
        self.verbosity = verbosity
        self.random_state = random_state

    def _iterator(self, X, y):
        """ Simple iterator function to batch the data """
        rng = check_random_state(self.random_state)
        if self.shuffle:
            rand_perm = rng.permutation(len(X))
            X = X[rand_perm]
            y = y[rand_perm]
        num_examples = len(X)
        num_batches = num_examples // self.batch_size
        for i in range(num_batches):
            sub_X = tr.from_numpy(X[i * self.batch_size:(i+1) * self.batch_size])
            sub_y = tr.from_numpy(y[i * self.batch_size:(i+1) * self.batch_size])
            yield sub_X, sub_y
        if num_examples % self.batch_size != 0:
            sub_X = tr.from_numpy(X[num_batches * self.batch_size:])
            sub_y = tr.from_numpy(y[num_batches * self.batch_size:])
            yield sub_X, sub_y
        return

    def _train_model(self, model, optimizer, X, y, X_test=None, y_test=None, callback=None):
        """ Trains the provided model using the provided data pairs X, y """
        # Setup output logging
        def default_callback(model, step, train_loss, train_acc, test_loss=None, test_acc=None, time_since_last=None, **kwargs):
            if self.verbosity >= 1:
                if test_acc is not None:
                    print(f"Step {step:4d} Train Loss {train_loss} Train Accuracy {train_acc} "
                          f"Test Loss {test_loss} Test Accuracy {test_acc}")
                else:
                    print(f"Step {step:4d} Train Loss {train_loss} Train Acc. {train_acc} "
                          f"Time {time_since_last}")
            elif self.verbosity > 0:
                print(f'S{step:4d} ', end='')
        if callback is None:
            callback = default_callback
        
        step = 0
        start_time = time.time()
        step_start_time = time.time()
        if self.verbosity > 0:
            print(' ') # Just print a new line
        while step < self.train_steps:
            for feat, label in self._iterator(X, y):
                if self.use_cuda:
                    feat = feat.cuda()
                    label = label.cuda()
                optimizer.zero_grad()
                output = model(feat)
                n_single = output.size(1)
                # Compute losses for all single index classifiers
                if self.use_cuda:
                    label_mat = tr.ger(label, tr.ones(n_single).cuda())
                else:
                    label_mat = tr.ger(label, tr.ones(n_single))
                losses = _cross_entropy_losses(
                    output, label_mat, self.cross_entropy_eps)
                loss = losses.sum()
                loss.backward()
                optimizer.step()
                step += 1
                # Show some results (could be simplified)
                if step % 100 == 0:
                    def get_stats(output, loss, label):
                        class_prob = output.cpu().data.numpy()
                        correct = np.rint(class_prob) == label.cpu().data.numpy()
                        accuracy = np.mean(correct)
                        loss = loss.cpu().data.numpy()
                        return loss, accuracy

                    all_results = np.array([
                        get_stats(o, l, label)
                        for o,l in zip(output.transpose(0,1), losses)
                    ])
                    train_loss = np.array([a[0] for a in all_results])
                    train_acc = np.array([a[1] for a in all_results])

                    if X_test is not None:
                        feat_test = tr.Tensor(X_test)
                        label_test = tr.Tensor(y_test)
                        if self.use_cuda:
                            feat_test = feat_test.cuda()
                            label_test = label_test.cuda()
                        output_test = model(feat_test)
                        losses_test = _cross_entropy_losses(
                            output_test, tr.ger(label_test, tr.ones(n_single).cuda()), self.cross_entropy_eps)
                        all_results_test = np.array([
                            get_stats(o, l, label_test)
                            for o, l in zip(output_test.transpose(0,1), losses_test)
                        ])
                        test_loss = np.array([a[0] for a in all_results])
                        test_acc = np.array([a[1] for a in all_results])
                    time_since_last = time.time() - step_start_time
                    step_start_time = time.time()
                    callback(**locals())
        total_time = time.time() - start_time
        if self.verbosity >= 1:
            print(f'Total training time: {total_time} seconds')
        elif self.verbosity > 0:
            print(' ')

    def fit(self, X, y=None, X_test=None, y_test=None):
        """ First makes a copy of the data X, y, and then fits it."""
        X, y = check_X_y(X, y, allow_nd=True, copy=True)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        classes = label_encoder.classes_
        X = X.astype(np.float32)  # Needed for PyTorch
        y = y.astype(np.float32)
        input_dim = X.shape[1]
        if X_test is not None:
            X_test, y_test = check_X_y(X_test, y_test, allow_nd=True, copy=True)
            X_test = X_test.astype(np.float32)  # Needed for PyTorch
            y_test = y_test.astype(np.float32)
            
        # Carefully seed training process (BEFORE torch model initialization)
        if self.random_state is not None:
            rng = check_random_state(self.random_state)
            seed = rng.randint(2**30)
            tr.manual_seed(seed)

        # Get model and optimizer
        model = copy.deepcopy(self.model) # copy model to avoid changing input
        if model is None:
            model = SingleIndexEnsembleModel(input_dim)
        def _default_get_optimizer(params):
            #return optim.SGD(params, lr=1e-3, momentum=0.9)
            return optim.Adam(params)
        get_optimizer = self.get_optimizer if self.get_optimizer is not None else _default_get_optimizer
        optimizer = get_optimizer(model.parameters())
        if self.use_cuda:
            model.cuda()
            
        self._train_model(
            model, optimizer, X, y, 
            X_test=X_test, y_test=y_test, callback=self.train_callback)
        # Convert to evaluation mode (required if model had dropout or batchnorm layers)
        model.eval()
        # Save parameters
        self.model_ = model
        self.label_encoder_ = label_encoder
        self.classes_ = classes
        self.rotation_ = self.model_.get_W_numpy(self.use_cuda)
        return self
    
    def predict_proba(self, X):
        # Note that output corresponds to Pr(y=1|x) rather than Pr(y=0|x)
        X = check_array(X).astype(np.float32)
        prob_1 = self.model_(tr.from_numpy(X)).detach().numpy()
        return np.array([1-prob_1, prob_1]).T

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]


class SingleIndexEnsembleModel(nn.Module):
    def __init__(self, input_dim, n_single=None, num_hidden=2, width=100):
        super().__init__()
        self.input_dim = input_dim
        if n_single is None:
            n_single = input_dim
        self.n_single = n_single
        self.num_hidden = num_hidden
        self.width = width
        Q, _ = tr.qr(tr.randn((input_dim, input_dim)))
        self.A = nn.Parameter(Q[:n_single, :])
        
        width_arr = list(width*np.ones((num_hidden+2), dtype=int))
        width_arr[0] = 1
        width_arr[-1] = 1
        self.width_arr = width_arr
        
        for i, (in_width, out_width) in enumerate(zip(width_arr[:-1], width_arr[1:])):
            # xA.T + b (like nn.linear)
            # input_dim is like a "batch" dimension in this case
            A = tr.zeros((n_single, out_width, in_width))
            b = tr.zeros((n_single, out_width))
            # Init of from nn.linear
            init.kaiming_uniform_(A, a=math.sqrt(5))
            #https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
            #fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            fan_in = A.size(0)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(b, -bound, bound)
            setattr(self, "A_%d" % i, nn.Parameter(A))
            setattr(self, "b_%d" % i, nn.Parameter(b))

    def forward(self, X):
        """ Returns the predicted probability given an input vector """
        X = self._forward_linear(X)
        X = X[:, :self.n_single] # Only keep n_single dimensions
        X = self._forward_univariate(X)
        # Note that we do not sum over features since this is ensemble 
        #  rather than GAM model (GAM would sum over features before sigmoid)
        X = tr.sigmoid(X)
        return X

    def _forward_linear(self, X):
        # Use housholder reflections
        def householder(X, u):
            # Use norm squared rather than normalizing u ahead of time
            norm_sq = tr.sum(u * u)
            utXt = tr.matmul(X, u)
            return X - (2/norm_sq) * tr.ger(utXt, u)
        for a in self.A:
            X = householder(X, a)
        return X

    def _forward_univariate(self, X):
        """ Returns the predicted probability given a single index value """
        # Get the right dimensions for matmult (N x K x 1)
        #print('N x K', X.size())
        X = X.view(X.size(0), X.size(1), 1, 1)
        #print('N x K x 1 x 1', X.size())
        for i, (in_width, out_width) in enumerate(zip(self.width_arr[:-1], self.width_arr[1:])):
            A = getattr(self, "A_%d" % i)
            b = getattr(self, "b_%d" % i)
            #transpose last two dimensions
            # X is (N x K x 1 x in), A is (K x out x in), A.T = (K x in x out)
            # out should be (N x K x 1 x out)
            #print('A: K x out x in', A.size())
            At = A.transpose(1, 2)
            #print('At: K x in x out', At.size())
            #print('X: N x K x 1 x in', X.size())
            X = tr.matmul(X, At)
            #print('X: N x K x 1 x out', X.size())
            bu = b.unsqueeze(1)
            #print('b(un): K x 1 x out', bu.size())
            X = X + bu
            #print('X: N x K x 1 x out', X.size())
            if i < (len(self.width_arr) - 1):
                #X = tr.relu(X)
                X = F.leaky_relu(X)
        # Take back down to (N x K)
        X = X.view(X.size(0), X.size(1))
        return X
            
    def get_W_numpy(self, use_cuda=False):
        # Note that we multiply H.T x = (x.T H).T 
        #  since we are multiplying housholders from the right
        eye = tr.eye(self.input_dim)
        if use_cuda:
            eye = eye.cuda()
        Wt_tensor = self._forward_linear(eye)
        Wt = Wt_tensor.detach().cpu().numpy()
        # absolute tolerance of 1e-6 since single precision arithmetic
        assert np.allclose(Wt @ Wt.T, np.eye(Wt.shape[0]), atol=1e-6), 'Not orthogonal'
        return Wt.T


def _cross_entropy_losses(pred_prob, labels, eps=1e-7):
    """ Returns the cross entropy loss given the predicted probs
    and the labels """
    loss = -tr.log(pred_prob + eps) * labels
    loss += -tr.log(1 - pred_prob + eps) * (1.0 - labels)
    return tr.mean(loss, dim=0)
