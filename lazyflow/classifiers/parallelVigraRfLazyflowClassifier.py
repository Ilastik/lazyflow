import os
import tempfile
from functools import partial
import cPickle as pickle
import collections

import numpy
import vigra
import h5py
import random

from lazyflow.utility import Timer
from lazyflow.request import Request, RequestPool, RequestLock
from lazyflowClassifier import LazyflowVectorwiseClassifierABC, LazyflowVectorwiseClassifierFactoryABC

import logging
logger = logging.getLogger(__name__)


class ParallelVigraRfLazyflowClassifierFactory(LazyflowVectorwiseClassifierFactoryABC):
    VERSION = 2 # This is used to determine compatibility of pickled classifier factories.
                # You must bump this if any instance members are added/removed/renamed.
                
    named_importances = None
    
    @staticmethod
    def setNamedImportances(new_named_importances):
        ParallelVigraRfLazyflowClassifierFactory.named_importances = new_named_importances
        
    @staticmethod
    def getNamedImportances():
        return ParallelVigraRfLazyflowClassifierFactory.named_importances 
    
    def __init__(self, num_trees_total=100, num_forests=None, variable_importance_path=None, label_proportion=None, variable_importance_enabled=True, **kwargs):      
        """
        num_trees_total: The number of trees to train
        num_forests: How many forests in which to distribute the trees (forests can train and predict in parallel)
                     If not provided, the number of forests is automatically determined 
                     to match the number of available lazyflow worker threads.
        kwargs: Additional keyword args, passed directly to the vigra.RandomForest constructor.
        """
                
        self._num_trees = num_trees_total
        self._label_proportion = label_proportion
        self._variable_importance_path = variable_importance_path
        self._variable_importance_enabled = variable_importance_enabled
        self._kwargs = kwargs
        
        # By default, num_forests matches the number of lazyflow worker threads
        self._num_forests = num_forests or Request.global_thread_pool.num_workers
        self._num_forests = max(1, self._num_forests)
    
    def set_num_trees(self, num_trees_total):
        self._num_trees = num_trees_total
        
    def set_variable_importance_path(self, variable_importance_path):
        self._variable_importance_path = variable_importance_path

    def set_label_proportion(self, label_proportion):
        self._label_proportion = label_proportion
        
    def create_and_train(self, X, y, feature_names=None):           
        # Distribute trees as evenly as possible
        tree_counts = numpy.array( [self._num_trees // self._num_forests] * self._num_forests )
        tree_counts[:self._num_trees % self._num_forests] += 1
        assert tree_counts.sum() == self._num_trees
        tree_counts = map(int, tree_counts)
        tree_counts[:] = (tree_count for tree_count in tree_counts if tree_count != 0)
        
        logger.debug( "Training parallel vigra RF" )
        # Save for future reference
        known_labels = numpy.unique(y)

        X = numpy.asarray(X, numpy.float32)
        y = numpy.asarray(y, numpy.uint32)
        if y.ndim == 1:
            y = y[:, numpy.newaxis]

        assert X.ndim == 2
        assert len(X) == len(y)

        # Create N forests
        forests = []
        for tree_count in tree_counts:
            forests.append( vigra.learning.RandomForest(tree_count, **self._kwargs) )

        # Train forests in parallel
        oobs = [None] * len(forests)
        importances = [None] * len(forests)
        
        def store_training_results(i, training_results):
            oob, importance_results = training_results
            oobs[i] = oob
            importances[i] = importance_results

        def store_oob_results(i, oob):
            oobs[i] = oob

        # Sample X and y
        if self._label_proportion:
            proportion = self._label_proportion
            row_num = int(proportion*X.shape[0])
            idx = random.sample(range(X.shape[0]), row_num)
            X = X[idx,:]
            y = y[idx] 
        
        # Train classifier with feature importance visitor    
        if self._variable_importance_enabled:
            if not feature_names:
                num_features = X.shape[1]
                feature_names = ["feature-{:02d}".format(i) for i in range(num_features)]
            
            with Timer() as train_timer:
                pool = RequestPool()
                for i, forest in enumerate(forests):
                    req = Request( partial(forest.learnRFWithFeatureSelection, X, y) )
                    # save the training results
                    req.notify_finished( partial( store_training_results, i ) )
                    pool.add( req )
                pool.wait()
                
            logger.info("Training took, {} ".format( train_timer.seconds() ) )    
    
            weights = numpy.array(tree_counts).astype(float)
            weights /= weights.sum()
    
            named_importances = collections.OrderedDict( zip( feature_names, numpy.average(importances, weights=weights, axis=0) ) )
            sorted_named_importances = collections.OrderedDict( sorted( named_importances.items(), key=lambda (k,v): v[-2], reverse=True) )  
            self.setNamedImportances(sorted_named_importances)
            
            importance_table = self._generate_importance_table( named_importances, sort=True )
            
            logger.info("Feature importance measurements during training: \n{}".format(importance_table) )  
            
        # train classifier without feature importance visitor       
        else:
            known_labels = None
            feature_names = None
            
            with Timer() as train_timer:
                pool = RequestPool()
                for i, forest in enumerate(forests):
                    req = Request( partial(forest.learnRF, X, y) )
                    # save the oob results
                    req.notify_finished( partial( store_oob_results, i ) )
                    pool.add( req )
                pool.wait()  
                       
        logger.info( "Training complete. Average OOB: {}".format( numpy.average(oobs) ) )
        return ParallelVigraRfLazyflowClassifier( forests, oobs, known_labels, feature_names )


    def _generate_importance_table(self, named_importances_dict, sort=True):
        """
        Return a string of the given importances dict, in csv format, 
        but also with extra spaces for pretty-printing.
        """
        import csv
        from StringIO import StringIO

        CSV_FORMAT = { 'delimiter' : ',', 'lineterminator' : '\n' }

        feature_name_length = max( map(len, named_importances_dict.keys()) )

        # See vigra/random_forest/rf_visitors.hxx, class VariableImportanceVisitor
        n_classes = len(named_importances_dict.values()[0]) - 2
        columns = [ "{: <{width}}".format("Feature Name", width=feature_name_length) ]
        columns += [ "  Class #{}".format(i) for i in range(n_classes)]
        columns += [ "   Overall" ]
        columns += [ "      Gini" ]
        
        output = StringIO()
        csv_writer = csv.writer(output, **CSV_FORMAT)
        csv_writer.writerow( columns )

        if sort:
            # Sort by "overall" importance (column -2)
            sorted_importances = sorted( named_importances_dict.items(),
                                         key=lambda (k,v): v[-2] )
            named_importances_dict = collections.OrderedDict( sorted_importances )

        for feature_name, importances in named_importances_dict.items():
            feature_name = "{: <{width}}".format(feature_name, width=feature_name_length)
            importance_strings = map( lambda x: "{: .07f}".format(x), importances )
            importance_strings = map( lambda s: "{: >10}".format(s), importance_strings )
            csv_writer.writerow( [feature_name] + importance_strings )
        
        # Save variable importance table to file
        if self._variable_importance_path :   
            file = open(os.path.join(self._variable_importance_path, 'varimp.txt'), 'w')
            file.write(output.getvalue())
            file.close()    
            
        return output.getvalue()

    def estimated_ram_usage_per_requested_predictionchannel(self):
        return (Request.global_thread_pool.num_workers) * 4

    @property
    def description(self):
        return "Parallel Vigra Random Forest Factory ({} trees total)"\
               .format( self._num_trees )

    def __eq__(self, other):
        return (    isinstance(other, type(self))
                and self._num_trees == other._num_trees
                and self._kwargs == other._kwargs )
    def __ne__(self, other):
        return not self.__eq__(other)

assert issubclass( ParallelVigraRfLazyflowClassifierFactory, LazyflowVectorwiseClassifierFactoryABC )

class ParallelVigraRfLazyflowClassifier(LazyflowVectorwiseClassifierABC):
    """
    Adapt the vigra RandomForest class to the interface lazyflow expects.
    """
    def __init__(self, forests, oobs, known_labels, feature_names=None):
        self._known_labels = known_labels
        self._forests = forests
        self._feature_names = feature_names
        
        # Note that oobs may not be in the same order as the forests.
        self._oobs = oobs
        
        self._num_trees = sum( forest.treeCount() for forest in self._forests )
    
    def predict_probabilities(self, X):
        logger.debug( "Predicting with parallel vigra RF" )
        X = numpy.asarray(X, dtype=numpy.float32)

        # As each forest completes, aggregate results in a shared array.
        # (Must put in a list so we can update it in this closure.)
        total_predictions = [None]
        prediction_lock = RequestLock()
        def update_predictions(forest, forest_predictions):
            forest_predictions *= forest.treeCount()
            with prediction_lock:
                if total_predictions[0] is None:
                    total_predictions[0] = forest_predictions
                else:
                    total_predictions[0] += forest_predictions

        # Create a request for each forest
        pool = RequestPool()
        for forest in self._forests:
            req = Request( partial( forest.predictProbabilities, X ) )
            req.notify_finished( partial(update_predictions, forest) )
            pool.add( req )
        del req
        pool.wait()

        total_predictions[0] /= self._num_trees
        return total_predictions[0]
    
    @property
    def oobs(self):
        return self._oobs
    
    @property
    def known_classes(self):
        return self._known_labels

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def feature_count(self):
        return self._forests[0].featureCount()

    def serialize_hdf5(self, h5py_group):
        for forest in self._forests:
            if forest is None:
                return

        name = h5py_group.name.split('/')[-1]
        # Due to non-shared hdf5 dlls, vigra can't write directly to
        # our open hdf5 group. Instead, we'll use vigra to write the
        # classifier to a temporary file.
        tmpDir = tempfile.mkdtemp()
        cachePath = os.path.join(tmpDir, 'tmp_classifier_cache.h5').replace('\\', '/')
        for i, forest in enumerate(self._forests):
            targetname = '{0}/{1}'.format(name, "Forest{:04d}".format(i))
            forest.writeHDF5(cachePath, targetname)

        parent_group = h5py_group.parent
        del parent_group[name]
        # Open the temp file and copy to our project group
        with h5py.File(cachePath, 'r') as cacheFile:
            parent_group.copy(cacheFile[name], name)

        h5py_group = parent_group[name]
        h5py_group['known_labels'] = self._known_labels
        if self._feature_names:
            h5py_group['feature_names'] = self._feature_names
        
        # This field is required for all classifiers
        h5py_group['pickled_type'] = pickle.dumps( type(self) )

        os.remove(cachePath)
        os.rmdir(tmpDir)
    
    @classmethod
    def deserialize_hdf5(cls, h5py_group):
        name = h5py_group.name.split('/')[-1]
        # Due to non-shared hdf5 dlls, vigra can't read directly
        # from our open hdf5 group. Instead, we'll copy the
        # classfier data to a temporary file and give it to vigra.
        tmpDir = tempfile.mkdtemp()
        cachePath = os.path.join(tmpDir, 'tmp_classifier_cache.h5').replace('\\', '/')
        with h5py.File(cachePath, 'w') as cacheFile:
            cacheFile.copy(h5py_group, name)

        forests = []
        for dset_name, forestGroup in sorted(h5py_group.items()):
            if dset_name.startswith('Forest'):
                targetname = '{0}/{1}'.format(name, dset_name)
                forests.append(vigra.learning.RandomForest(cachePath, targetname))

        try:
            known_labels = list(h5py_group['known_labels'][:])
        except KeyError:
            # Older projects didn't store the labels explicitly.
            known_labels = range(1, forests[0].labelCount()+1 )

        try:
            feature_names = list(h5py_group['feature_names'][:])
        except KeyError:
            # Older projects don't store feature names.
            feature_names = None

        try:
            oobs = list(h5py_group['oobs'][:])
        except KeyError:
            # Older projects didn't store the oobs.
            # Just provide something obviously invalid.
            oobs = [-1.0] * len(forests)

        os.remove(cachePath)
        os.rmdir(tmpDir)

        return ParallelVigraRfLazyflowClassifier( forests, oobs, known_labels, feature_names )

assert issubclass( ParallelVigraRfLazyflowClassifier, LazyflowVectorwiseClassifierABC )
