from pycar.raicar import RAICAR
from pycar.runlogger import Logger
from pycar.utilities import standardize
import numpy as np
import glob,cPickle

class TestRAICARrpy2ica:
    
    def setup(self):
        filePtr = open('data/icatestsignals.db','r')
        signalDict = cPickle.load(filePtr)
        filePtr.close()
        # setup the test data
        self.signals = np.vstack([signalDict[k] for k in signalDict])
        # standardize signals
        self.signals = standardize(self.signals,stdtype='row')
        # mixing
        self.A = np.random.randn(len(self.signals),len(self.signals))
        self.X = np.dot(self.A,self.signals)
        # ica options
        icaOptions = dict()
        icaOptions['nonlinearity'] = 'logcosh'
        icaOptions['maxIterations'] = 500
        icaOptions['tolerance'] = 1.0e-05
        # this is just to get the setup to run; test_has_ica will fail if rpy2ica isn't there
        # other tests may pass if you have pyica installed, however
        try:
            from rpy2ica import fastica
            icaMethod = fastica
        except:
            icaMethod = None
        # RAICAR object
        self.K = 10
        # data logger
        self.logger = Logger('runlog.txt')
        # raicar object
        self.mycar = RAICAR(projDirectory='raicartest',K=self.K,nSignals=3,icaOptions=icaOptions,icaMethod=icaMethod)


    def test_has_ica(self):
        try:
            import rpy2ica
            rpy2ica_loaded = True
        except ImportError:
            rpy2ica_loaded = False
        assert rpy2ica_loaded,"rpy2ica not installed; you need to use pyica.fastica or supply your own icaMethod and icaOptions!"

    
    def test_clean_project(self):
        self.mycar.clean_project()
        assert len(glob.glob('raicartest/ica/*.h5')) == 0,"project not cleaned succesfully: aligned components exist"
        assert len(glob.glob('raicartest/aln/*.aln')) == 0,"project not cleaned successfully: ica realizations exist"
        assert len(glob.glob('raicartest/rab/*.db')) == 0,"project not cleaned successfully: R(a,b) matrices exist"
    

    def test_run_ica(self):
        self.mycar.kica(self.X)
        assert len(glob.glob('raicartest/ica/*.h5')) == self.K,"ica failed: proper number of realizations not produced"

    
    def test_compute_rab(self):
        self.mycar.kica(self.X)
        self.mycar.compute_rab()
        assert(len(self.mycar.RabDict)) == (self.K*(self.K-1))/2,"problem with computing Rab: incorrect number of cross-realization matrices"
        maxCorr = 0.0
        minCorr = 1.0
        for k in self.mycar.RabDict:
            matrixMax = self.mycar.RabDict[k].max()
            matrixMin = self.mycar.RabDict[k].min()
            if matrixMax >= maxCorr:
                maxCorr = matrixMax
            if matrixMin <= minCorr:
                minCorr = matrixMin
        assert minCorr >= 0.0,"problem with Rab matrices : minimum absolute correlation is < 0.0"
        assert maxCorr <= 1.0,"problem with Rab matrices : maximum absolute correlation is > 1.0"

    
    def test_compute_component_alignments(self):
        self.mycar.kica(self.X)
        self.mycar.compute_rab()
        self.mycar.compute_component_alignments()
        assert len(self.mycar.alignDict) == 3,"problem with computed alignments: wrong number of RAICAR components"
        assert len(self.mycar.alignDict[0]) == self.K,"problem with computed alignments: wrong number of realization components in RAICAR component"

    def test_align_components(self):
        self.mycar.kica(self.X)
        self.mycar.compute_rab()
        self.mycar.compute_component_alignments()
        for k in xrange(3):
            self.mycar.align_component(k)
        assert len(glob.glob('raicartest/aln/*.h5')) == 3,"problem with aligned components: not enough components"
        assert len(glob.glob('raicartest/aln/*.db')) == 1,"problem with aligned components: alignment database was not created"

    def test_construct_raicar_components(self):
        self.mycar.kica(self.X)
        self.mycar.compute_rab()
        self.mycar.compute_component_alignments()
        for k in xrange(3):
            self.mycar.align_component(k)
        self.mycar.construct_raicar_components()
        assert np.min(self.mycar.reproducibility) >= 0,"problem with component reproducibility: R < 0"
        assert np.max(self.mycar.reproducibility) <= 1,"problem with component reproducibility: R > 1"


class TestRAICARpyica:

    def setup(self):
        filePtr = open('data/icatestsignals.db','r')
        signalDict = cPickle.load(filePtr)
        filePtr.close()
        # setup the test data
        self.signals = np.vstack([signalDict[k] for k in signalDict])
        # standardize signals
        self.signals = standardize(self.signals,stdtype='row')
        # mixing
        self.A = np.random.randn(len(self.signals),len(self.signals))
        self.X = np.dot(self.A,self.signals)
        # ica options
        icaOptions = dict()
        icaOptions['algorithm'] = 'parallel fp'
        icaOptions['nonlinearity'] = 'logcosh'
        icaOptions['decorrelation'] = 'mdum'
        icaOptions['maxIterations'] = 500
        icaOptions['tolerance'] = 1.0e-05
        icaOptions['alpha'] = 1.0
        # this is just to get the setup to run; test_has_ica will fail if pyica.fastica isn't there

        try:
            from pyica import fastica
            icaMethod = fastica
        except:
            icaMethod = None
        # RAICAR object
        self.K = 10
        # data logger
        self.logger = Logger('runlog.txt')
        # raicar object
        self.mycar = RAICAR(projDirectory='raicartest',K=self.K,nSignals=3,icaOptions=icaOptions,icaMethod=icaMethod)


    def test_has_ica(self):
        try:
            import pyica
            pyica_loaded = True
        except ImportError:
            pyica_loaded = False
        assert pyica_loaded, "pyica not installed; you need to use rpy2ica.fastica or supply your own icaMethod and icaOptions!"

    
    def test_clean_project(self):
        self.mycar.clean_project()
        assert len(glob.glob('raicartest/ica/*.h5')) == 0,"project not cleaned succesfully: aligned components exist"
        assert len(glob.glob('raicartest/aln/*.aln')) == 0,"project not cleaned successfully: ica realizations exist"
        assert len(glob.glob('raicartest/rab/*.db')) == 0,"project not cleaned successfully: R(a,b) matrices exist"
    

    def test_run_ica(self):
        self.mycar.kica(self.X)
        assert len(glob.glob('raicartest/ica/*.h5')) == self.K,"ica failed: proper number of realizations not produced"

    
    def test_compute_rab(self):
        self.mycar.kica(self.X)
        self.mycar.compute_rab()
        assert(len(self.mycar.RabDict)) == (self.K*(self.K-1))/2,"problem with computing Rab: incorrect number of cross-realization matrices"
        maxCorr = 0.0
        minCorr = 1.0
        for k in self.mycar.RabDict:
            matrixMax = self.mycar.RabDict[k].max()
            matrixMin = self.mycar.RabDict[k].min()
            if matrixMax >= maxCorr:
                maxCorr = matrixMax
            if matrixMin <= minCorr:
                minCorr = matrixMin
        assert minCorr >= 0.0,"problem with Rab matrices : minimum absolute correlation is < 0.0"
        assert maxCorr <= 1.0,"problem with Rab matrices : maximum absolute correlation is > 1.0"

    
    def test_compute_component_alignments(self):
        self.mycar.kica(self.X)
        self.mycar.compute_rab()
        self.mycar.compute_component_alignments()
        assert len(self.mycar.alignDict) == 3,"problem with computed alignments: wrong number of RAICAR components"
        assert len(self.mycar.alignDict[0]) == self.K,"problem with computed alignments: wrong number of realization components in RAICAR component"

    def test_align_components(self):
        self.mycar.kica(self.X)
        self.mycar.compute_rab()
        self.mycar.compute_component_alignments()
        for k in xrange(3):
            self.mycar.align_component(k)
        assert len(glob.glob('raicartest/aln/*.h5')) == 3,"problem with aligned components: not enough components"
        assert len(glob.glob('raicartest/aln/*.db')) == 1,"problem with aligned components: alignment database was not created"

    def test_construct_raicar_components(self):
        self.mycar.kica(self.X)
        self.mycar.compute_rab()
        self.mycar.compute_component_alignments()
        for k in xrange(3):
            self.mycar.align_component(k)
        self.mycar.construct_raicar_components()
        assert np.min(self.mycar.reproducibility) >= 0,"problem with component reproducibility: R < 0"
        assert np.max(self.mycar.reproducibility) <= 1,"problem with component reproducibility: R > 1"