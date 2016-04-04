import math
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class rb_plot_curves:
    
    def __init__(self):
        pass
        
    def plot_validation_curve(self, x_train, y_train, x_test, y_test, data_label, learner, param_name, param_range, saveName, legend_loc='lower right', stat_name='rmse', use_param_range=None, use_sk=False): 
        
        if use_param_range == None:
            use_param_range = param_range
            
        train_scores = np.ones([param_range.shape[0], 1])
        test_scores = np.ones([param_range.shape[0], 1])
        
        for i in range(param_range.shape[0]):
            param_value = param_range[i]
            setattr(learner, param_name, param_value)
            
            if use_sk:
                learner.fit(x_train, y_train)
                predY = learner.predict(x_train)
            else:
                learner.addEvidence(x_train, y_train)
                predY = learner.query(x_train)
                
            if stat_name == 'rmse':
                stat = math.sqrt(((y_train - predY) ** 2).sum()/y_train.shape[0])
            else:
                stat = np.corrcoef(predY, y=y_train)[0,1]
            train_scores[i,0] = stat
            
            if use_sk:
                predY = learner.predict(x_test)
            else:
                predY = learner.query(x_test)
                
            
            if stat_name == 'rmse':
                stat = math.sqrt(((y_test - predY) ** 2).sum()/y_test.shape[0])
            else:
                stat = np.corrcoef(predY, y=y_test)[0,1]
            test_scores[i,0] = stat
            
            
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        
        plt.clf()
        plt.cla()
        plt.plot(use_param_range, train_mean,
                 color='blue', marker='o',
                 markersize=5,
                 label='in sample')
        
        plt.plot(use_param_range, test_mean,
                 color='green', marker='s',
                 markersize=5, linestyle='--',
                 label='out of sample')
        
        plt.grid()
        plt.title("Overfitting: %s" % (data_label))
        plt.xlabel(param_name)
        plt.ylabel('Error')
        plt.legend(loc=legend_loc)
        plt.savefig(saveName)
        
    def plot_predicted_actual(self, Y, predY, desc, datasetName, saveName):
        plt.clf()
        plt.cla()
        
        N = Y.shape[0]
        colors = np.random.rand(N)
        # random
        area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses
        
        plt.cla()
        plt.clf()
        
        fig, ax = plt.subplots()
        ax.scatter(Y, predY, s=area, c=colors, alpha=0.5)
        ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4, c='magenta')
        plt.title("Compare %s: %s" % (desc, datasetName))
        plt.grid(True)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        plt.savefig(saveName)
        