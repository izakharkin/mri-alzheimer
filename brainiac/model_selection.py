from sklearn.model_selection import StratifiedKFold
import numpy as np


class ManualGroupKFold():
    """ K-fold iterator variant with non-overlapping groups.
        The same group will not appear in two different folds (the number of
        distinct groups has to be at least equal to the number of folds).
        The folds are approximately balanced in the sense that the number of
        distinct targets is approximately the same in each fold.
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
    random_state : None, int
        Pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.
    Example 
    -------
    >>> target = np.array([1]*10+ [0]*10)
    >>> groups = np.array([i//2 for i in range(20)])
    >>> X = np.random.random((20,3))
    >>> mgf = ManualGroupKFold(n_splits = 3, random_state = 52)
    >>> print('Target {}, Groups {}'.format(target, groups))
    >>> for train, test in mgf.split(X, target, groups):
    ...    print('-----------------------------------------')
    ...    print('Train : {}, Test : {}'.format(train, test))
    ...    print('Target train : {}, Target test : {}'.format(target[train], target[test]))
    ...    print('Groups train : {}, Groups test : {}'.format(groups[train], groups[test]))
    Target [1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0], Groups [0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9]
    -----------------------------------------
    Train : [ 0  1  6  7  8  9 12 13 16 17 18 19], Test : [ 2  3  4  5 10 11 14 15]
    Target train : [1 1 1 1 1 1 0 0 0 0 0 0], Target test : [1 1 1 1 0 0 0 0]
    Groups train : [0 0 3 3 4 4 6 6 8 8 9 9], Groups test : [1 1 2 2 5 5 7 7]
    -----------------------------------------
    Train : [ 2  3  4  5  8  9 10 11 14 15 18 19], Test : [ 0  1  6  7 12 13 16 17]
    Target train : [1 1 1 1 1 1 0 0 0 0 0 0], Target test : [1 1 1 1 0 0 0 0]
    Groups train : [1 1 2 2 4 4 5 5 7 7 9 9], Groups test : [0 0 3 3 6 6 8 8]
    -----------------------------------------
    Train : [ 0  1  2  3  4  5  6  7 10 11 12 13 14 15 16 17], Test : [ 8  9 18 19]
    Target train : [1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0], Target test : [1 1 0 0]
    Groups train : [0 0 1 1 2 2 3 3 5 5 6 6 7 7 8 8], Groups test : [4 4 9 9]
    """
    def __init__(self, n_splits = 3, random_state = None):

        self.n_splits = n_splits
        self.random_state = random_state
        
    def get_n_splits(self, X, y, groups):

        return self.n_splits
    
    def split(self, X, target, groups):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : numpy ndarray
            of shape (object, features) data object
        target : numpy ndarray
            of shape (object, ) target variable,
            folds are approximately balanced by this variable
        groups : numpy ndarray
            of shape (object, ) characteristic variable,
            objects from the same group will occur in the same fold
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        skf = StratifiedKFold(n_splits = self.n_splits,
                             shuffle = True, random_state = self.random_state)
        target_unique = np.array([target[groups == elem][0] for elem in np.unique(groups)])
        names_unique = np.unique(groups)
        idx = np.arange(X.shape[0])
               
        for train, test in skf.split(np.zeros(target_unique.shape[0]), target_unique):
            
            train_labels = np.array(names_unique)[train]
            test_labels = np.array(names_unique)[test]
            train_idx = np.in1d(groups, train_labels)
            test_idx = np.in1d(groups, test_labels)
            
            yield idx[train_idx], idx[test_idx]