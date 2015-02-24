from pycar.bicar import BICAR
from pycar.bicar import BICARTransferFunction,lag_gamma
from pycar.utilities import standardize
import numpy as np
import glob,cPickle

class TestBICARrpy2ica:

    def setup(self):
        # load BLOBS test data
        self.testData = cPickle.load(open('tests/bicartestdata.db','rb'))
        # TF for unmixing
        invIRF = lag_gamma
        invPar = {'alpha':1.0,'tau':0.15,'t0':0.1}
        self.transfer = BICARTransferFunction(invIRF,invPar,self.testData['slowSamp'],self.testData['dt'])
        # number of signals
        self.nSignals = 5
        # ica options
        icaOptions = dict()
        icaOptions['nonlinearity'] = 'logcosh'
        icaOptions['maxIterations'] = 500
        icaOptions['tolerance'] = 1.0e-05
        # ensures setup runs; presence of ica is tested elsewhere
        try:
            from rpy2ica import fastica
            icaMethod = fastica
        except:
            icaMethod = None
        # create bicar object
        self.mycar = BICAR(projDirectory='tests/rpy2icabicartest',nSignals=self.nSignals,K=30,reportLevel=0,icaOptions=icaOptions,icaMethod=icaMethod)
    
    def test_run_bicar(self):
        self.mycar.clean_project()
        self.mycar.runall(self.testData['X'],self.testData['YT'],self.transfer,similarity='abspearson',degenerate=False)
        repro = self.mycar.read_reproducibility()
        assert len(repro) == self.nSignals, "Wrong number of BICAR components extracted!"
        assert np.min(repro) > 0.0, "Negative reproducibility values are present!"
        assert np.max(repro) <= 1.0, "Reproducibility values greater than unity present!"

    def test_has_ica(self):
        try:
            import rpy2ica
            rpy2ica_loaded = True
        except ImportError:
            rpy2ica_loaded = False
        assert rpy2ica_loaded,"rpy2ica not installed; you need to use pyica.fastica or supply your own icaMethod and icaOptions!"

class TestBICARpyica:

    def setup(self):
        # load BLOBS test data
        self.testData = cPickle.load(open('tests/bicartestdata.db','rb'))
        # TF for unmixing
        invIRF = lag_gamma
        invPar = {'alpha':1.0,'tau':0.15,'t0':0.1}
        self.transfer = BICARTransferFunction(invIRF,invPar,self.testData['slowSamp'],self.testData['dt'])
        # number of signals
        self.nSignals = 5
        # ica options
        icaOptions = dict()
        icaOptions['nonlinearity'] = 'logcosh'
        icaOptions['maxIterations'] = 500
        icaOptions['tolerance'] = 1.0e-05
        # ensures setup runs; presence of ica is tested elsewhere
        try:
            from pyica import fastica
            icaMethod = fastica
        except:
            icaMethod = None
        # create bicar object
        self.mycar = BICAR(projDirectory='tests/pyicabicartest',nSignals=self.nSignals,K=30,reportLevel=0,icaOptions=icaOptions,icaMethod=icaMethod)

    def test_run_bicar(self):
        self.mycar.clean_project()
        self.mycar.runall(self.testData['X'],self.testData['YT'],self.transfer,similarity='abspearson',degenerate=False)
        repro = self.mycar.read_reproducibility()
        assert len(repro) == self.nSignals, "Wrong number of BICAR components extracted!"
        assert np.min(repro) > 0.0, "Negative reproducibility values are present!"
        assert np.max(repro) <= 1.0, "Reproducibility values greater than unity present!"

    def test_has_ica(self):
        try:
            import pyica
            pyica_loaded = True
        except ImportError:
            pyica_loaded = False
        assert pyica_loaded,"pyica not installed; you need to use rpy2ica.fastica or supply your own icaMethod and icaOptions!"