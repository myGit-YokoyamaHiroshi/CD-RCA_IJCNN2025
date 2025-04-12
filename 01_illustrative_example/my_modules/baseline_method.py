# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 01:04:18 2024
"""
import numpy as np
import scipy as sp
from sklearn import linear_model
from tigramite import data_processing as pp

def LIME(xtest, ytest, model, pred_target, var_names, 
         N_grad=100, eta=0.01, l1=1e-5, seed=None):

    xtest    = xtest.ravel()
    M        = len(xtest) 
    
    if seed is not None:
        rng      = np.random.default_rng(seed)
        Xlocal   = rng.normal(loc=xtest,scale=eta,size=(N_grad,M))
    else:
        Xlocal   = np.random.normal(loc=xtest,scale=eta,size=(N_grad,M))
        
    new_data = pp.DataFrame(Xlocal, var_names = var_names)
    y_pred   = model.predict(pred_target, new_data=new_data)
    
    flocal   = y_pred - ytest
    
    fitLasso = linear_model.Lasso(alpha=l1,fit_intercept=True)
    
    fitLasso.fit(Xlocal[:-model.tau_max,:], flocal)
    grad = fitLasso.coef_
    intercept = fitLasso.intercept_

    return grad, intercept

def z_score(xtest,Xtrain,eps=1.e-10):

    xmean = Xtrain.mean(axis=0)
    sigma = Xtrain.std(axis=0)
    score = (xtest-xmean)/(sigma + eps)
    return score

def EIG_vec_i(idx, x_test, model, Xtrain, 
              pred_target, var_names, tau_max, 
              N_alpha =100, N_grad=10,
             eta=0.1, h_minimum = 1e-8, seed=0):

    import numpy as np

    # Input must be a Numpy array. No list, no int as the input.
    x_test = np.array(x_test).astype(float).ravel()

    # Grid points for the trapezoidal integral of the alpha parameter of IG.
    # The grit points has numbers from 0 through N_alpha 
    # (i.e., the total grid points is N_alpha +1). 
    dal = 1./N_alpha
    alphas = dal*np.arange(0,N_alpha+1)

    # Baseline position = training data. Hence, IG is NOT doubly black-box.
    X_base = Xtrain
    N_base, M = X_base.shape

    # Pre-populate perturbations for gradient estimation
    if seed is not None:
        h0 = np.random.default_rng(seed).normal(0,eta,N_grad)
    else:
        h0 = np.random.normal(0,eta,N_grad)
        
    h = h0[abs(h0) >= h_minimum*eta] # Drop too small perturbations. 
    NN_grad = len(h) ## Use this instead of N_grad. NN_grad can be smaller than N_grad.
    

    #------ The rightmost term in the trapezoid rule except for the (-1/2) prefactor.
    # This term dees not depend on x_base.
    # (x^t-x_base) term 
    termN_A = - (X_base[:,idx] - x_test[idx]).mean(axis=0) 
    
    # Gradient term
    Z_right    = np.tile(x_test.reshape(1,-1),[NN_grad,1])
    Z_right_pp = pp.DataFrame(Z_right, var_names = var_names)
    f0_right   = model.predict(pred_target, new_data=Z_right_pp)
    # f0_right = model.predict(Z_right)
    
    Z_right[:,idx] = Z_right[:,idx] + h
    Z_right_pp     = pp.DataFrame(Z_right, var_names = var_names)
    f_right        = model.predict(pred_target, new_data=Z_right_pp)
    # f_right = model.predict(Z_right)

    termN = termN_A * dal*( (f_right - f0_right)/h[tau_max:] ).sum()/(NN_grad-tau_max)


    #----- The leftmost term except for the (-1/2) prefactor. 
    # Does depend on x_base.
    # (x^t-x_base) term
    dx = (x_test[idx] - X_base[:,idx]).reshape(-1,1) # Column vector
    term0_A = np.tile(dx,[NN_grad,1]).flatten(order='F')/N_base

    # Gradient term
    Z_left    = np.tile(X_base, [NN_grad,1])
    Z_left_pp = pp.DataFrame(Z_left, var_names = var_names)
    f0_left   = model.predict(pred_target, new_data=Z_left_pp)
    # f0_left = model.predict(Z_left)
    
    
    hh_left = np.tile(h.reshape(1,-1),[N_base,1]).flatten(order='F')
    Z_left[:,idx] = Z_left[:,idx] + hh_left
    Z_left_pp     = pp.DataFrame(Z_left, var_names = var_names)
    f_left        = model.predict(pred_target, new_data=Z_left_pp)
    # f_left = model.predict(Z_left)
    
    term0_B = dal*( (f_left - f0_left)/hh_left[tau_max:])/NN_grad
    term0   = (term0_A[tau_max:] * term0_B).sum()

    #--- For non-terminal points ---

    # (x^t-x_base) term
    dx = (x_test[idx] - X_base[:,idx]).reshape(1,-1) # row vector
    dX = np.tile(dx, [NN_grad*(N_alpha+1),1]) # tall matrix stacking the row vectors
    term_A = dX.flatten(order='F')/N_base # Stacking the tall matrix vertically

    ZB = np.empty([(N_alpha+1)*NN_grad*N_base, M])
    for n in range(N_base):
        x_base = X_base[n,:]
        Zn = alphas.reshape(-1,1).dot(x_test.reshape(1,-1)) \
            + (1 - alphas).reshape(-1,1).dot(x_base.reshape(1,-1))
        ZBn = np.tile(Zn,[NN_grad,1])

        i1 = n*(NN_grad*(N_alpha +1))
        i2 = i1 + NN_grad*(N_alpha+1)
        ZB[i1:i2,:] = ZBn

    # Output before perturbation ---
    ZB_pp = pp.DataFrame(ZB, var_names = var_names)
    fB0   = model.predict(pred_target, new_data=ZB_pp)
    # fB0 = model.predict(ZB)
    
    # Output after perturbation ----
    # Computing perturbation vector
    hh = np.tile(h.reshape(1,-1),[N_alpha+1,1]).flatten(order='F')
    hhB = np.tile(hh.reshape(-1,1),[N_base,1]).ravel()
    # Computing f(x) with perturbation
    ZB[:,idx] = ZB[:,idx] + hhB
    ZB_pp = pp.DataFrame(ZB, var_names = var_names)
    fB    = model.predict(pred_target, new_data=ZB_pp)
    # fB = model.predict(ZB)
    # term_B
    term_B = (dal/NN_grad) * ((fB-fB0)/hhB[tau_max:])

    term = (term_A[tau_max:]*term_B).sum()


    #-------- EIG 
    EIG_i = term -0.5*(term0 + termN)

    return EIG_i


def EIG_vec(x_test, model, Xtrain, 
            pred_target, var_names, tau_max, 
            N_alpha =100, N_grad=10,
            eta=0.1, h_minimum = 1e-8, seed=0):
    x_test = np.array(x_test).astype(float).ravel()

    X_base = Xtrain
    N_base, M = X_base.shape

    EIG = np.empty(M)
    for ii in range(M):
        EIG[ii] = EIG_vec_i(ii, x_test, model, Xtrain, 
                            pred_target, var_names, tau_max,
                            N_alpha =N_alpha, N_grad=N_grad, 
                            eta=eta, h_minimum = h_minimum, 
                            seed=seed)
    return EIG

#%%%%%%%%%%%%%%%%%%%%
def gpa_map(X, y, model, a, b, var_names, pred_target,
            l2=0.1, l1_ratio=0.5,
            lr = 0.05, lr_shrinkage = 0.9,
            itr_max=50, RAE_th=1e-3, reporting_interval='auto',
            N_grad=10, eta_stddev = 0.1, delta_init_scale = 1e-8,
            seed_initialize=1,
            seed_grad = 1, h_minimum=1.e-8, verbose=False):
    '''
    Finding a MAP solution for Generative Perturbation Analysis (GPA).

    Parameters
    ----------
    X : 2D ndarray
        Row-based data matrix for the input variables. Typically, X has only
        one row. Multiple rows are for collective attribution.
    y : list-like
        y values corresponding to the rows of X
    model : TYPE
        black-box regression function object with a predict() implemented.
    a : float
        The shape parameter of the gamma prior. 2a corresponds to 
        the degrees of freedom of the resulting t-distribution.
    b : float
        The rate parameter of the gamma prior. Can be an array rather than a scalar. 
    l2 : float, optional
        The precision parameter of the Gaussian prior. The default is 0.1.
    l1_ratio : float, optional
        The L1 strength of the elastic net prior, defined relative to the l2 strength. 
        l2*l1_ratio gives the l1 regularization strength. The default is 0.5.
    lr : float, optional
        The learning rate kappa. The default is 0.05.
    lr_shrinkage : float, optional
        The shrinkage rate of the learning rate (see Ide et al AAAI 21). 
        The default is 0.9.
    itr_max : int, optional
        The maximum number of iteration. The default is 50.
    RAE_th : float, optional
        The threshold for the relative absolute error. Used to judge the convergence. 
        The default is 1e-3.
    reporting_interval : int, optional
        If this is 10, errors are reported every 10 iterations. The default is 'auto',
        which uses reporting_interval = int(itr_max/10). 
    N_grad : int, optional
        The number of perturbations for gradient estimation. The default is 10.
    eta_stddev : float, optional
        Standard deviation of Gaussian used for Monte Carlo gradient estimation.
        The default is 0.1.
    delta_init_scale : float, optional
        The scale of random small noise for initializing the MAP value of delta 
        The default is 1e-8.
    seed_initialize : int, optional
        Random seed for initializing delta. The default is 1.
    seed_grad : int, optional
        Random seed for Monte Carlo gradient estimation. The default is 1.
    h_minimum : float, optional
        Minimum threshold of the absolute increment to avoid divide-by-zero. 
        Generated increments below this threshold will be discarded. The default is 1.e-8.
    verbose : boolean, optional
        Set True if you want detailed updates in the course of iteration. The default is False.

    Returns
    -------
    delta : array
        The MAP estimate of the attribution score delta.
    params : dict
        gradients : array
            Each row is the gradient for each test sample.
        obj_values : float
            The objective value (log likelihood)
        itr : int
            The number of iteration upon finishing the loop. 
        lr_final : float
            The final value of the learning rate. Note that learning rate gets smaller geometrically. 
        RAEs_delta : 
            Relative absolute error of delta of each iteration
        RAEs_objec : 
            RAE of the objective value of delta of each iteration

    '''

    if reporting_interval == 'auto':
        reporting_interval = int(itr_max/10)
    if X.ndim == 1:
        X= X.reshape(1,-1)
        y= np.array([y])

    N_test,M = X.shape

    # Constant term of the log likelihood ##### J(delta)-dependent #####
    gam_ratio = np.log( sp.special.gamma(a+0.5)/sp.special.gamma(a) )
    const_obj = 0.5*N_test*np.log(2*np.pi) \
        + 0.5*(np.log(b)).sum() + N_test*gam_ratio \
            -0.5*M*np.log(l2) +0.5*M*np.log(2.*np.pi)

    # Initializing delta -----------------------------------
    rng = np.random.default_rng(seed_initialize)
    delta_initial = rng.normal(0,delta_init_scale,M)
    delta_l1_initial = np.max([np.abs(delta_initial).sum(),delta_init_scale])
    delta = delta_initial.copy()

    delta_old = np.repeat(np.Inf, M)
    objective_old = -np.Inf

    # Allocating memory
    gradients = np.empty([N_test,M]) # local gradient in each row
    g_vec = np.empty(M) # The g vector of the prox grad
    Z = np.empty(X.shape) # The data matrix that keeps getting updated. 
    DeltaN = np.empty(N_test)

    # Iterative updates start -----------------------
    obj_values = []
    RAEs_delta = []
    RAEs_objec = []

    for itr in range(itr_max):

        Z[:,:] = X + delta.reshape(1,-1)

        # Computing y - f(x+delta) 
        Zpp    = pp.DataFrame(Z, var_names = var_names)
        y_pred = model.predict(pred_target, new_data=Zpp)
        DeltaN[:] = y - y_pred

        # Computing local gradients (stored in each row)
        gradients[:,:] = local_gradient_vec2(Z=Z, model=model, N_grad=N_grad,
                                             var_names=var_names, pred_target=pred_target, 
                                             eta_stddev=eta_stddev,
                                             seed = seed_grad,
                                             h_minimum=h_minimum)

        # Computing the g vector. This is a 1D array. ##### J(delta)-dependent #####
        g_vec[:] = ( gradients*( (DeltaN/(2*b + DeltaN**2)\
                                 ).reshape(-1,1)) ).sum(axis=0)
        g_vec[:] = (1-lr*l2)*delta + lr*(2*a+1)*g_vec

        # The lasso solution with the L1 term (proximal gradient algorithm)
        delta[:] = prox_l1(g_vec,lr*l2*l1_ratio)

        # Computing the objective function ##### J(delta)-dependent #####
        obj_value = (a+0.5)*(np.log(1 + (DeltaN**2/(2*b))) ).sum() \
            + 0.5*l2*(delta**2).sum() + const_obj
        obj_values = obj_values + [obj_value]

        # Checking convergence
        delta_L1norm = np.max([np.abs(delta).sum(),delta_l1_initial])
        RAE_delta = np.abs(delta - delta_old).sum()/delta_L1norm
        RAEs_delta = RAEs_delta + [RAE_delta]

        RAE_objec = np.abs(obj_value - objective_old)/np.abs(obj_value)
        RAEs_objec = RAEs_objec + [RAE_objec]

        if (RAE_delta  <= RAE_th) and (RAE_objec <= RAE_th):
            break

        # Prepping for the next round
        delta_old[:] = delta[:]
        objective_old = obj_value
        lr = lr*lr_shrinkage

        # Reporting
        if ((itr+1)%reporting_interval == 0) and verbose:
            print(f'{itr+1:4d}: RAE(d)={RAE_delta:f},',end='')
            print(f'RAE(o)={RAE_objec:f}, obj_value={obj_value}')

    print(f'finished:itr={itr+1:4d}: RAE(d)={RAE_delta:f},',end='')
    print(f'RAE(o)={RAE_objec:f}, obj_value={obj_value}')

    params = {'gradients':gradients, 'obj_values':np.array(obj_values),
              'itr':(itr+1), 'lr_final':lr,
              'RAEs_delta':RAEs_delta, 'RAEs_objec':RAEs_objec}
    return delta, params


def gpa_map_gaussian(X, y, model, stddev_yf, var_names, pred_target,
                     l2=0.1, l1_ratio=0.5, lr = 0.05, lr_shrinkage = 0.9,
                     itr_max=50, RAE_th=1e-3, reporting_interval='auto',
                     N_grad=10, eta_stddev = 0.1, delta_init_scale = 1e-8, seed_initialize=1,
                     seed_grad = 1, h_minimum=1.e-8, verbose=False):
    '''
    Computes the MAP value of delta based on (Gaussian observation)
    +(elastic net prior) model. This should return the same score as
    Likelihood Compensation (LC; Ide et al. AAAI 21).

    Uses local_gradient_vec2() for gradient estimation.

    Parameters
    ----------
    X : array-like
        Test data matrix of test input. If N_test = 1, X becomes 1-row matrix.
    y : array
        Test data output.
    model : TYPE
        Black-box regression model object that allows .predict().
    stddev_yf : float
        Predictive standard deviation. This is NOT the variance. 
    l2 : float, optional
        L2 regularization strength or the precision of the Gaussian prior。The default is 0.1.
    l1_ratio : TYPE, optional
        L1 regularization strength relative to that of L2. The default is 0.5.
    lr : float, optional
        Learning rate kappa. The default is 0.05.
    lr_shrinkage : float, optional
        lr gets shrunken by lr = lr*lr_shrinkage. The default is 0.9.
    itr_max : int, optional
        Maximum number of iterations. The default is 50.
    RAE_th : float, optional
        Threshold of convergence of the relative absolute error of delta and 
        the negative log likelihood. The default is 1e-3.
    reporting_interval : int, optional
        How often you want to get updates. The default is 'auto', which uses
        reporting_interval = int(itr_max/10).
    N_grad : int, optional
        The number of random perturbations for local gradient estimation. The default is 10.
    eta_stddev : TYPE, optional
        The standard deviation used in Monte Carlo gradient estimation. The default is 0.1.
    delta_init_scale : TYPE, optional
        The scale of delta for random initialization. The default is 1e-8.
    seed_initialize : int, optional
        Random seed of initialization of delta. The default is 1.
    seed_grad : TYPE, optional
        Random seed of random perturbation in gradient estimation. The default is 1.
    h_minimum : TYPE, optional
        The minimum threshold of absolute perturbation, below which the perturbation is discarded. 
        This is to avoid divide-by-zero. The default is 1.e-8.
    verbose : boolean, optional
        If True, print errors. The default is False.

    Returns
    -------
    delta : ndarray
        The value of delta (LC).
    params : dict
        gradients : array
            Each row is the gradient for each test sample.
        obj_values : float
            The objective value (log likelihood)
        itr : int
            The number of iteration upon finishing the loop. 
        lr_final : float
            The final value of the learning rate. Note that learning rate gets smaller geometrically. 
        RAEs_delta : 
            Relative absolute error of delta of each iteration
        RAEs_objec : 
            RAE of the objective value of delta of each iteration

    '''

    if reporting_interval == 'auto':
        reporting_interval = int(itr_max/10)
    if X.ndim == 1:
        X= X.reshape(1,-1)
        y= np.array([y])

    # Verifying input
    N_test,M = X.shape
    if type(stddev_yf) is not np.ndarray:
        local_lambda = 1/np.repeat(stddev_yf**2, N_test)

    # Constant term of the negative log likelihood to be minimized. ##### J(delta)-dependent #####
    const_obj = 0.5*M*np.log(2.*np.pi) -0.5*M*np.log(l2) \
        +0.5*N_test*np.log(2*np.pi) - 0.5*np.log(local_lambda).sum()

    # Initializing delta -----------------------------------
    if seed_initialize != None:
        rng = np.random.default_rng(seed_initialize)
        delta_initial = rng.normal(0,delta_init_scale,M)
    else:
        delta_initial = np.random.normal(0,delta_init_scale,M)
        
    delta_l1_initial = np.max([np.abs(delta_initial).sum(),delta_init_scale])
    delta = delta_initial.copy()

    delta_old = np.repeat(np.Inf, M)
    objective_old = -np.Inf

    # Assigning memory space
    gradients = np.empty([N_test,M]) # local gradient in the row.
    g_vec = np.empty(M) # The g vector of prox grad 
    Z = np.empty(X.shape) # The data matrix that keeps getting updated
    DeltaN = np.empty(N_test)

    # Iteration starts -----------------------------
    obj_values = []
    RAEs_delta = []
    RAEs_objec = []

    for itr in range(itr_max):

        Z[:,:] = X + delta.reshape(1,-1)

        # Computing y - f(x+delta) 
        
        Zpp    = pp.DataFrame(Z, var_names = var_names)
        y_pred = model.predict(pred_target, new_data=Zpp)
        DeltaN[:] = y - y_pred#model.predict(Z)

        # Computing the gradient that is stored in the rows for each test sample. 
        gradients[:,:] = local_gradient_vec2(Z=Z, model=model, N_grad=N_grad,
                                             var_names=var_names, pred_target=pred_target, 
                                             eta_stddev=eta_stddev,
                                             seed= seed_grad, h_minimum=h_minimum)

        # Computing the g vector (1D array) ##### J(delta)-dependent #####
        g_vec[:] = (gradients*((DeltaN*local_lambda).reshape(-1,1))).sum(axis=0)
        g_vec[:] = (1 - lr*l2)*delta + lr*g_vec

        # Prox grad with L1 regularization
        delta[:] = prox_l1(g_vec,lr*l2*l1_ratio)

        # Computing the objective function ##### J(delta)-dependent #####
        obj_value = 0.5*((DeltaN**2)*local_lambda).sum() \
            + 0.5*l2*(delta**2).sum() + const_obj
        obj_values = obj_values + [obj_value]

        # Checking convergence
        delta_L1norm = np.max([np.abs(delta).sum(),delta_l1_initial])
        RAE_delta = np.abs(delta - delta_old).sum()/delta_L1norm
        RAEs_delta = RAEs_delta + [RAE_delta]

        RAE_objec = np.abs(obj_value - objective_old)/np.abs(obj_value)
        RAEs_objec = RAEs_objec + [RAE_objec]

        if (RAE_delta  <= RAE_th) and (RAE_objec <= RAE_th):
            break

        # Prepping for the next round
        delta_old[:] = delta[:]
        objective_old = obj_value
        lr = lr*lr_shrinkage

        # Reporting
        if ((itr+1)%reporting_interval == 0) and verbose:
            print(f'{itr+1:4d}: RAE(d)={RAE_delta:f},',end='')
            print(f'RAE(o)={RAE_objec:f}, obj_value={obj_value}')

    print(f'finished:itr={itr+1:4d}: RAE(d)={RAE_delta:f},',end='')
    print(f'RAE(o)={RAE_objec:f}, obj_value={obj_value}')

    params = {'gradients':gradients, 'obj_values':np.array(obj_values),
              'itr':(itr+1), 'lr_final':lr,
              'RAEs_delta':RAEs_delta, 'RAEs_objec':RAEs_objec}

    return delta, params



def local_gradient_vec2(Z, model,N_grad, var_names, pred_target,
                        eta_stddev=0.1, seed=1,h_minimum=1.e-8):
    '''
    Given a black-box function f(x), simultaneously computes the local gradient
    at each of the rows of the data matrix Z.

    If you have 5 samples of 10-dimensional vectors, you will get a gradient
    matrix of the same size: 5x10 matrix, where each row is the gradient
    at the corresponding sample.

    Note that `model` is a black-box function. Unlike autograd functions in
    deep learning frameworks, we do not need any analytic form of the model.

    Parameters
    ----------
    Z : 2D ndarray of float
        data matrix whose rows are the coordinates at which gradient is evaluated.
    model : TYPE
        An object representing a black-box function. `predict()` has to be available.
    N_grad : int
        The number of perturbations generated. Typically, this is 10.
    eta_stddev : float, optional
        Standard deviation of the perturbations. The default is 0.1.
    seed : int, optional
        Seed for random perturbations. The default is 1.
    h_minimum : float, optional
        Allowable smallest scale of the perturbation. Note that the actual
        minimum threshold is given by h_minimum*eta_stddev.
        The default is 1.e-8.

    Returns
    -------
    grad_matrix : 2D ndarray
        Matrix of the gradients in the rows.

    '''

    N_test, M = Z.shape
    grad_matrix = np.zeros([N_test,M])

    # hh ベクトルを作る
    if seed == None:
        h0 = np.random.normal(0,eta_stddev,N_grad)
    else:
        h0 = np.random.default_rng(seed).normal(0,eta_stddev,N_grad)

    h = h0[abs(h0) >= h_minimum*eta_stddev] # Discard too small perturbations.
    NN_grad = len(h) ## Use this instead of N_grad. If there is any discarded perturbation, 
    # NN_grad will be smaller than the original N_grad. 
    hh = np.zeros([N_test,NN_grad]) + h
    hh = hh.flatten(order='F')

    # Creating the long hhh vector to leverage the vectorized computation capability of numpy. 
    hhh = np.tile(hh.reshape(-1,1),[M,1])

    # Creating huge data matrix
    ZB0 = np.tile(Z,[NN_grad*M,1])
    ZB  = ZB0.copy()

    # Evaluate the function value. Just one time.
    ZB0pp = pp.DataFrame(ZB0, var_names = var_names)
    ff0   = model.predict(pred_target, new_data=ZB0pp)#model.predict(ZB0)

    # Concatenating perturbed data matrices. Further concatenate them M times. 
    # Then fill values to them. The only difference is that the idx-th column is perturbed. 
    for idx in range(M):
        n_start = N_test*NN_grad*idx
        n_end = n_start + N_test*NN_grad
        ZB[n_start:n_end,idx] = ZB0[n_start:n_end,idx] + hh

    # Evaluate the function value. Just one time. 
    ZBpp = pp.DataFrame(ZB, var_names = var_names)
    ff   = model.predict(pred_target, new_data=ZBpp)
    # ff = model.predict(ZB)
    df = (ff-ff0)/hhh.ravel()[1:]

    for idx in range(M-model.tau_max):
        n_start = N_test*NN_grad*idx 
        n_end = n_start + N_test*NN_grad 
        ge =  df[n_start:n_end].reshape([N_test,N_grad],order='F')
        grad_matrix[:,idx] =ge.mean(axis=1)

    return grad_matrix


def prox_l1(phi,mu):
    '''
    Proximal operator for L1 regularization.
    prox(phi | mu||phi||)

    Parameters
    ----------
    phi : 1D ndarray
        Input argument of the L1 prox operator.
    mu : float
        L1 regularization strength. Note: when used in the proximal gradient
        method, you need to multiply the l1 strength by the learning rate.

    Returns
    -------
    phi : 1D ndarray
        Regularized solution, i.e., prox(phi | mu||phi||).

    '''
    mask1 = (abs(phi) <= mu)
    mask2 = (phi < (-1)*mu)
    mask3 = (phi > mu)
    phi[mask1] = 0
    phi[mask2] = phi[mask2] + mu
    phi[mask3] = phi[mask3] - mu
    return phi
