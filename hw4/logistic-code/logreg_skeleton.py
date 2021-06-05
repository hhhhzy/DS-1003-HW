def f_objective(theta, X, y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    num_instances = X.shape[0]
    num_features = X.shape[1]   
    sum_risk = 0   
    for i in range(num_instances):
        sum_risk += np.logaddexp(0, -y[i]*np.dot(theta,X[i]))
    
    risk = sum_risk/num_instances
    reg = l2_param * np.dot(theta,theta)
    J = risk + reg
    return J
    
    
def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
        