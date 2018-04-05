import GPy
import numpy as np

from models.abstractmodel import AbstractModel


class GPlib(AbstractModel):
    # The training_targets are the Y's which are real numbers
    def __init__(self, 
                 training_data,
                 training_targets,
                 model_params):
        
        self.data_mean = np.mean(training_data, axis=0)
        self.data_std = np.std(training_data, axis=0) + 1e-10
	self.training_data = (training_data - self.data_mean) / self.data_std
	self.pseudo_points = 0
	
        self.targets_mean = np.mean(training_targets, axis=0)
        self.targets_std = np.std(training_targets, axis=0) + 1e-10
	self.training_targets = (training_targets - self.targets_mean) / self.targets_std        

	self.n_points = training_data.shape[0]
        self.input_d = training_data.shape[1]
        self.output_d = training_targets.shape[1]
        
        self.kern = model_params.kern
        self.optimizer_restarts = model_params.optimizer_restarts

        self.models = []
        self.train()


    def train(self):
        self.models = []
        for i in range(0, self.output_d):
            if self.kern == 'matern':
                kernel = GPy.kern.Matern52(input_dim=self.input_d, ARD=True)  
            elif self.kern == 'rbf':
                kernel = GPy.kern.RBF(input_dim=self.input_d, ARD=True)  
            else:
                raise Exception('Ill-defined kernel')

            model = GPy.models.GPRegression(self.training_data, self.training_targets[:, i:i+1], kernel)
            model.optimize_restarts(num_restarts=self.optimizer_restarts, robust=True)
            model.optimize(messages=False)
            self.models.append(model)

    def addPseudoPoint(self, x):
	self.training_data = np.vstack((x, self.training_data))
	self.training_targets = np.vstack((np.zeros((self.output_d)), self.training_targets))
        for i, model in enumerate(self.models):
            self.training_targets[-1:, i:i+1], vars = model.predict(np.reshape(x, (1, -1)), full_cov=False)
	    model.set_XY(self.training_data, self.training_targets[:, i:i+1])
	self.pseudo_points += 1

    def removePseudoPoints(self):
	self.training_data = self.training_data[:-self.pseudo_points, :]
	self.training_targets = self.training_targets[:-self.pseudo_points, :]
        for i, model in enumerate(self.models):
	    model.set_XY(self.training_data, self.training_targets[:, i:i+1])
	self.pseudo_points = 0

    def addPoint(self, x, y):
        assert self.pseudo_points == 0
	self.training_data = np.vstack(((x - self.data_mean) / self.data_std, self.training_data))
        self.training_targets = np.vstack(((y - self.targets_mean) / self.targets_std, self.training_targets))
        self.train()

    def predictBatch(self, test_data, samples):
        means = np.zeros((samples, test_data.shape[0], self.output_d))
        vars = np.zeros((samples, test_data.shape[0], self.output_d))
        for i in range(samples):
            for j, model in enumerate(self.models):
                means[i, :, j:j+1], vars[i, :, j:j+1] = model.predict(test_data, full_cov=False)

        return means * self.targets_std[None, None, :] + self.targets_mean[None, None, :], vars * np.square(self.targets_std[None, None, :]) 
