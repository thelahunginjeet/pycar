'''
Created on June 9, 2011

BICAR is a module for fusing temporal and spatial ICA data (assumed to represent different measurement modalities
of a common source process).  This algorithm is a modified dual-RAICAR with a matching step that associates
temporal sources with spatial loadings, using a supplied transfer function and downsampling/interpolation, 
assuming a convolutive (LTI) model.

@author: Kevin S. Brown, University of Connecticut

This source code is provided under the BSD-3 license, duplicated as follows:

Copyright (c) 2013, Kevin S. Brown
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this 
list of conditions and the following disclaimer in the documentation and/or other 
materials provided with the distribution.

3. Neither the name of the University of Connecticut  nor the names of its contributors 
may be used to endorse or promote products derived from this software without specific 
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER 
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import scipy as sp
import tables as tb
import unittest,cPickle,sys,os,gc,glob

from random import choice as rchoice
from scipy import signal,corrcoef,histogram,histogram2d,special
from scipy.stats import pearsonr,spearmanr,kendalltau

# other pycar dependencies
from raicar import *
from utilities import empirical_ci, standardize, corrmatrix, construct_file_name, deconstruct_file_name

# for mi calculation, if you want to pair sources that way
def binned_entropy(px):
    '''
    Computes the entropy from pre-binned data px (so px is a discrete probabiltity distribution,
    obtained by binning some numeric data).  Entropy is computed in nats.
    '''
    # dump zero bins to avoid log problems
    px = px[np.nonzero(px)]
    return -1.0*(px*np.log(px)).sum()


def entropy(x,bins=10):
    '''
    Shannon entropy, in nats, of a continuous (unbinned) set of data x.  This is a very simple 
    (but potentially poor) estimator that comes from naive binning.
    '''
    px = histogram(x,bins,density=True)[0]
    return binned_entropy(px)
    

def mutual_information(x,y,bins=10):
    '''
    Computes the mutual information between x and y.  Assumes x and y are unbinned data vectors.
    Only works if x and y are the same size.  This is a *very* naive estimator that is obtained
    by simply directly computing a 2d histogram for x and y.
    '''
    pxy = histogram2d(x.flatten(),y.flatten(),bins)[0]
    # fake the density=True flag that scipy.histogram has
    pxy = pxy/pxy.sum()
    return entropy(x,bins) + entropy(y,bins) - binned_entropy(pxy)


# possible impulse response functions
def single_gamma(t,alpha=8.6,tau=0.55):
    return (sp.power((t/tau),alpha)*np.exp(-t/tau))/(tau*special.gamma(alpha+1))

def lag_gamma(t,alpha=8.6,tau=0.55,t0=0.55):
    return 0.5*(1+np.sign(t-t0))*(sp.power(((t-t0)/tau),alpha)*np.exp(-(t-t0)/tau)).real/(tau*special.gamma(alpha+1))

def double_gamma(t,a1=12.0,tau1=0.5,a2=12.0,tau2=0.85,wt=0.5):
    return single_gamma(t,a1,tau1) - wt*single_gamma(t,a2,tau2)

# exceptions
class BICARICAException(Exception):
    def __init__(self):
        print "Number of extracted spatial sources depends on the number of temporal sources. Run temporal kica first, or supply the number of sources."
        
class BICARICATypeException(Exception):
    def __init__(self,t):
        print "Unknown ica type %s provided to kica." % t
        
class BICARMatchingException(Exception):
    def __init__(self):
        print "No temporal-spatial matching has yet been performed."

class BICARMatchingMethodException(Exception):
    def __init__(self,s):
        print "Unknown matching method %s." % s
        
class BICARBackgroundException(Exception):
    def __init__(self):
        print "No reproducibility background has been calculated."


class BICARTransferFunction(object):
    '''Used to filter and downsample an input signal (assumed to be from temporal ICA) in order
    to match it to the loadings from spatial ICA sources.
    
    Parameters:
        ------------
            irf : callable method, required
                function giving the impulse response function that associates the datasets
                
            parDict : dictionary, required
                parameters to be supplied to the irf
            
            matchSamp : list, required 
                a list of samples for matching the timebase of the signals from tICA and sICA

            dt : float, required
                1/R, where R (Hz) is the sampling rate of the high (temporal) resolution data.
                converts between samples and seconds
    ''' 
    def __init__(self,irf,parDict,matchSamp,dt):
        self.irf = irf
        self.p = parDict
        self.x = matchSamp
        self.dt = dt
        
    def compute_irf(self,t):
        '''Computes the supplied irf, passing through the parameters supplied to the constructor.'''
        return self.irf(t,**self.p)
    
    def filter(self,s):
        '''Accepts an input source - assumed to be from temporal ICA - and filters it using the
        supplied irf. Returns the convolved signal.'''
        # compute filter points out to about 25% of the maximum time (~0.25*dt*matchSamp[-1])
        maxt = 0.25*self.dt*self.x[-1]
        t = np.arange(0,maxt,self.dt)
        return sp.convolve(self.compute_irf(t),s,mode='full')[0:len(s)]
    
    def resample(self,s):
        return sp.interp(self.x,xrange(0,len(s)),s)
    
    def transform(self,s):
        '''Shorthand function that does s -> filter -> resample -> b.'''
        return self.resample(self.filter(s))
        

class BICAR(RAICAR):
    def __init__(self,projDirectory,nSignals=None,K=30,avgMethod='weighted',canonSigns=True,icaMethod=None,icaOptions=None,reportLevel=2):
        super(BICAR,self).__init__(projDirectory,nSignals,K,avgMethod,canonSigns,icaMethod,icaOptions)
        # BICAR-specific member data
        self.projDirectory = projDirectory
        self.temporalDirectory = os.path.join(self.projDirectory,'tICA')  # tICA realizations
        self.spatialDirectory = os.path.join(self.projDirectory,'sICA')      # sICA realizations
        self.matDirectory = os.path.join(self.projDirectory,'mat')          # component matching
        self.bkgDirectory = os.path.join(self.projDirectory,'bkg')          # uses for background calculations
        # dictionary of allowed similarity functions
        self.simFuncs = {'random' : self.similarity_random, 'abspearson' : self.similarity_abspearson, 'pwtpearson' : self.similarity_pwtpearson,
                         'absspearman': self.similarity_absspearman, 'pwtspearman' : self.similarity_pwtspearman, 'abskendall': self.similarity_abskendall,
                         'pwtkendall': self.similarity_pwtkendall, 'mi' : self.similarity_mi}
        # controls output verbosity:
        #    0 : print nothing
        #    1 : print only large-scale messages, not which files are being processed
        #    2 : print everything
        self.reportLevel = reportLevel
        # associates tICA components with sICA components:
        #    matchDict[n] = [(i1,j1,c1),...,(iK,jK,cK)] means for realization n, tICA comp i1 matches sICA comp j1, etc.,
        # and ci records the absolute correlation for the match
        self.matchDict = dict()
        # spatialAlignDict[j] = [d0,d1,...,dK-1], constructed using alignDict and the matchDict 
        self.spatialAlignDict = {}
        # will hold the eventual bicar sources/mixing matrices
        self.temporalSources = None
        self.temporalMixing = None
        self.spatialSources = None
        self.spatialMixing = None
    
    
    
    def spatial_alignment(self):
        '''
        Uses the corrected temporal alignment to obtain the corresponding alignment of spatial sources (which
        are associated to the temporal sources via the matchDict).  After this operation, the two alignment
        dictionaries should have the same form.
        '''
        if len(self.alignDict) == 0:
            try:
                alnPtr = open(os.path.join(self.alnDirectory,'alignment.db'),'rb')
                self.alignDict = cPickle.load(alnPtr)
                alnPtr.close()
            except:
                raise RAICARAlignmentException
        if len(self.matchDict) == 0:
            try:
                matPtr = open(os.path.join(self.matDirectory,'matching.db'),'rb')
                self.matchDict = cPickle.load(matPtr)
                matPtr.close()
            except:
                raise BICARMatchingException
        self.spatialAlignDict = dict().fromkeys(self.alignDict.keys())
        for k in self.alignDict:
            self.spatialAlignDict[k] = [self.matchDict[i][self.alignDict[k][i]][0] for i in xrange(0,len(self.alignDict[k]))]
        # pickle the matching result
        alnPtr = open(os.path.join(self.alnDirectory,'spatial_alignments.db'),'wb')
        cPickle.dump(self.spatialAlignDict,alnPtr,protocol=-1)
        alnPtr.close()
        
            
    def clean_project(self):
        '''
        Removes all files in the subdirectories of the project directory, as well as the directories.  
        Subdirectories which do not exist (having not yet been created) are skipped.
        '''
        projDirectories = [self.temporalDirectory,self.spatialDirectory,self.rabDirectory,self.alnDirectory,self.racDirectory]
        for d in projDirectories:
            if not os.path.exists(d):
                if self.reportLevel > 0:
                    print "Nothing to clean: directory %s does not exist" % d
            else:
                if self.reportLevel > 0:
                    print "Cleaning %s" % d
                files = os.listdir(d)
                for f in files:
                    os.remove(os.path.join(d,f))
                    
    
    def kica(self,X,icaType='temporal'):
        '''
        Accepts a data matrix X:
            X : nX x tX, tX >> nX  (icaType = 'temporal')
            Y : tY x nY, tY << nY  (icaType = 'spatial')
        and runs K ica realizations on X.  The number of requested sources for spatial ICA is set by the number 
        requested from temporal ICA (hence tICA must be run first).  Note that Y needs to be properly transposed (row dim << col dim)
        on input.
        '''
        if icaType == 'temporal':
            d = self.temporalDirectory
        else:
            d = self.spatialDirectory
        if not os.path.exists(d):
            try:
                os.mkdir(d)
            except OSError:
                pass
        # files to make
        icaToMake = [os.path.join(d,construct_file_name(icaType[0]+'ICARun',x,'h5')) for x in range(0,self.K)]
        if self.nSignals is None:
            if icaType == 'temporal':
                self.nSignals = X.shape[0]
            else:
                raise BICARICAException
        for f in icaToMake:
            if not os.path.exists(f):
                if self.reportLevel > 0:
                    print 'Running %s ICA realization %s' % (icaType,f)
                A,W,S = self.ica(X,nSources=self.nSignals,**self.icaOptions)
                # write the results to hdf5
                h5Ptr = tb.openFile(f,mode="w",title='ICA Realization')
                decomp = h5Ptr.createGroup(h5Ptr.root,'decomps','ICA Decomposition')
                h5Ptr.createArray(decomp,'sources',S,"S")
                h5Ptr.createArray(decomp,'mixing',A,"A")
                h5Ptr.close()
            else:
                if self.reportLevel > 0:
                    print 'ICA realization %s already exists.  Skipping.' % f
                

    def similarity_random(self,tftPtr,sfiPtr,transfer):
        '''
        Purely uniform random similarities transfer function is completely ignored. 
        Useful for tests of statistical significance.
        '''
        return np.random.rand(self.nSignals,self.nSignals)


    def similarity_abspearson(self,tfiPtr,sfiPtr,transfer):
        '''
        Measures similarity via absolute pearson correlation.
        '''
        Sij = np.zeros((self.nSignals,self.nSignals))
        bi = sfiPtr.getNode('/decomps/mixing').read()
        for k in xrange(self.nSignals):
            s = tfiPtr.root.decomps.sources[k,:]
            b = transfer.transform(s)
            # correlate b (the transformed source) with all the bi
            for l in xrange(self.nSignals):
                Sij[k,l] = np.abs(pearsonr(b,bi[:,l])[0])
        return Sij

    
    def similarity_pwtpearson(self,tfiPtr,sfiPtr,transfer):
        '''
        Measures similarity via absolute pearson correlation, weighted
        by 1 minus the asmptotic p-value of the correlation.
        '''
        Sij = np.zeros((self.nSignals,self.nSignals))
        bi = sfiPtr.getNode('/decomps/mixing').read()
        for k in xrange(self.nSignals):
            s = tfiPtr.root.decomps.sources[k,:]
            b = transfer.transform(s)
            # correlate b (the transformed source) with all the bi
            for l in xrange(self.nSignals):
                (r,p) = pearsonr(b,bi[:,l])
                Sij[k,l] = np.abs(r)*(1.0-p)
        return Sij


    def similarity_absspearman(self,tfiPtr,sfiPtr,transfer):
        '''
        Measures similarity via absolute spearman correlation.
        '''
        Sij = np.zeros((self.nSignals,self.nSignals))
        bi = sfiPtr.getNode('/decomps/mixing').read()
        for k in xrange(self.nSignals):
            s = tfiPtr.root.decomps.sources[k,:]
            b = transfer.transform(s)
            # correlate b (the transformed source) with all the bi
            for l in xrange(self.nSignals):
                Sij[k,l] = np.abs(spearmanr(b,bi[:,l])[0])
        return Sij


    def similarity_pwtspearman(self,tfiPtr,sfiPtr,transfer):
        '''
        Measures similarity via absolute spearman correlation, weighted
        by 1 minus the asmptotic p-value of the correlation.
        '''
        Sij = np.zeros((self.nSignals,self.nSignals))
        bi = sfiPtr.getNode('/decomps/mixing').read()
        for k in xrange(self.nSignals):
            s = tfiPtr.root.decomps.sources[k,:]
            b = transfer.transform(s)
            # correlate b (the transformed source) with all the bi
            for l in xrange(self.nSignals):
                (r,p) = spearmanr(b,bi[:,l])
                Sij[k,l] = np.abs(r)*(1.0-p)
        return Sij


    def similarity_abskendall(self,tfiPtr,sfiPtr,transfer):
        '''
        Measures similarity via absolute kendall tau correlation.
        '''
        Sij = np.zeros((self.nSignals,self.nSignals))
        bi = sfiPtr.getNode('/decomps/mixing').read()
        for k in xrange(self.nSignals):
            s = tfiPtr.root.decomps.sources[k,:]
            b = transfer.transform(s)
            # correlate b (the transformed source) with all the bi
            for l in xrange(self.nSignals):
                Sij[k,l] = np.abs(kendalltau(b,bi[:,l])[0])
        return Sij


    def similarity_pwtkendall(self,tfiPtr,sfiPtr,transfer):
        '''
        Measures similarity via absolute spearman correlation, weighted
        by 1 minus the asmptotic p-value of the correlation.
        '''
        Sij = np.zeros((self.nSignals,self.nSignals))
        bi = sfiPtr.getNode('/decomps/mixing').read()
        for k in xrange(self.nSignals):
            s = tfiPtr.root.decomps.sources[k,:]
            b = transfer.transform(s)
            # correlate b (the transformed source) with all the bi
            for l in xrange(self.nSignals):
                (r,p) = kendalltau(b,bi[:,l])
                Sij[k,l] = np.abs(r)*(1.0-p)
        return Sij

    
    def similarity_mi(self,tfiPtr,sfiPtr,transfer):
        '''
        Measures similarity via the mutual information between the temporal
        source and the spatial mixing matrix elements (time series).  Completely
        ignores the transfer function.
        '''
        Sij = np.zeros((self.nSignals,self.nSignals))
        bi = sfiPtr.getNode('/decomps/mixing').read()
        for k in xrange(self.nSignals):
            s = tfiPtr.root.decomps.sources[k,:]
            # compute the mi of the temporal source with all the bi
            for l in xrange(self.nSignals):
                # downsample s
                sdown = transfer.resample(s)
                Sij[k,l] = mutual_information(sdown,bi[:,l])
        return Sij


    def match_sources_degen(self,similarity,transfer):
        '''
        Allows degenerate (many tICA -> one sICA) matching.
        '''
        matchDict = dict()
        tICAFiles = sorted(os.listdir(self.temporalDirectory))
        if len(tICAFiles) == 0:
            raise RAICARICAException
        for tfi in tICAFiles:
            if self.reportLevel > 1:
                print 'Matching sources from %s' % tfi
            i = np.int(deconstruct_file_name(tfi)[1])
            matchDict[i] = dict()
            tfiPtr = tb.openFile(os.path.join(self.temporalDirectory,tfi),'r')
            # get the corresponding sICA file
            try:
                sfiPtr = tb.openFile(os.path.join(self.spatialDirectory,construct_file_name('sICARun',i,'h5')),'r')
            except:
                raise RAICARICAException
            # similarities computed here
            Sij = self.simFuncs[similarity](tfiPtr,sfiPtr,transfer)
            # just find all the row maxima, even if they occur in the same columns
            matchInd = Sij.argmax(axis=1)
            for k in xrange(self.nSignals):
                matchDict[i][k] = (matchInd[k],Sij[k,matchInd[k]])
            tfiPtr.close()
            sfiPtr.close()
        return matchDict


    def match_sources_nondegen(self,similarity,transfer):
        '''
        Forces nondegenerate (one tICA -> one sICA) matching.
        '''
        matchDict = dict()
        tICAFiles = sorted(os.listdir(self.temporalDirectory))
        if len(tICAFiles) == 0:
            raise RAICARICAException
        for tfi in tICAFiles:
            if self.reportLevel > 1:
                print 'Matching sources from %s' % tfi
            i = np.int(deconstruct_file_name(tfi)[1])
            matchDict[i] = dict()
            tfiPtr = tb.openFile(os.path.join(self.temporalDirectory,tfi),'r')
            # get the corresponding sICA file
            try:
                sfiPtr = tb.openFile(os.path.join(self.spatialDirectory,construct_file_name('sICARun',i,'h5')),'r')
            except:
                raise RAICARICAException
            # similarities computed here
            Sij = self.simFuncs[similarity](tfiPtr,sfiPtr,transfer)
            # we have the similiarity matrices.  search for successive maxima
            for k in xrange(self.nSignals):
                bigS = Sij.max()
                row,col = np.unravel_index(Sij.argmax(),Sij.shape)
                matchDict[i][row] = (col,bigS)
                # zero out the row/col where we matched the pair
                Sij[row,:] = 0.0
                Sij[:,col] = 0.0
            tfiPtr.close()
            sfiPtr.close()
        return matchDict
    
                
    def match_sources(self,similarity,transfer,degenerate=False):
        '''
        A dispatcher method to match temporal and spatial sources.  Basically just extra wrapping, but
        it takes care of directory creation and pickling of the result.
        '''
        if not os.path.exists(self.matDirectory):
            try:
                os.mkdir(self.matDirectory)
            except OSError:
                pass
        if degenerate:
            matchfunc = self.match_sources_degen
            matchmeth = 'many to one'
        else:
            matchfunc = self.match_sources_nondegen
            matchmeth = 'one to one'
        if self.reportLevel > 0:
            print 'Matching method : %s ' % matchmeth
            print 'Similarity measure : %s ' % similarity
        # compute the matching dictionary
        self.matchDict = matchfunc(similarity,transfer)
        # pickle the matching result, stored in the match dict
        matPtr = open(os.path.join(self.matDirectory,'matching.db'),'wb')
        cPickle.dump(self.matchDict,matPtr,protocol=-1)
        matPtr.close()
        
    
    def compute_rab(self):
        '''
        Uses the current set of ICA realizations (pytabled) to compute K*(K-1)/2 cross-correlation matrices;
        they are indexed via tuples.  R(a,b) is much smaller than the ICA realizations (all R(a,b) matrices 
        are generally smaller than ONE realization), so R(a,b) is also retained in memory. Recomputation of 
        the R(a,b) matrices is forced.  R(a,b) matrices from paired temporal/spatial sources are combined
        using:
            R(a,b) = 0.5*R_t(a,b) + 0.5*R_s(a,b)
        This assumes the number of samples (timepoints in the tICA sources and locations in the sICA sources)
        in the two datasets are comparable; otherwise a weighted sum should be used.         
        '''
        if not os.path.exists(self.rabDirectory):
            try:
                os.mkdir(self.rabDirectory)
            except OSError:
                pass
        if len(self.matchDict) == 0:
            try:
                matPtr = open(os.path.join(self.matDirectory,'matching.db'),'rb')
                self.matchDict = cPickle.load(matPtr)
                matPtr.close()
            except:
                raise BICARMatchingException
        if self.nSignals is None:
            # need the number of signals for matrix sizing, available from the matching dictionary
            self.nSignals = len(zip(*self.matchDict[0])[0])       
        # temporal files to loop over 
        tICAFiles = sorted(os.listdir(self.temporalDirectory))
        if len(tICAFiles) == 0:
            raise RAICARICAException
        for tif in tICAFiles:
            i = np.int(deconstruct_file_name(tif)[1])
            tiPtr = tb.openFile(os.path.join(self.temporalDirectory,tif),'r')
            if self.reportLevel > 1:
                print 'Working on R(%d,b)'%i
            try:
                siPtr = tb.openFile(os.path.join(self.spatialDirectory,construct_file_name('sICARun',i,'h5')),'r')
            except:
                raise RAICARICAException
            # used to link temporal and spatial sources
            for tjf in tICAFiles:
                j = np.int(deconstruct_file_name(tjf)[1])
                if j > i:
                    try:
                        sjPtr = tb.openFile(os.path.join(self.spatialDirectory,construct_file_name('sICARun',j,'h5')),'r')
                    except:
                        raise RAICARICAException
                    self.RabDict[(i,j)] = np.zeros((self.nSignals,self.nSignals))
                    # all sources assumed to have unit std. dev. but nonzero mean - will behave badly otherwise!
                    tjPtr = tb.openFile(os.path.join(self.temporalDirectory,tjf),'r')
                    # double loop over signals
                    for l in range(0,self.nSignals):
                        for m in range(0,self.nSignals):
                            # temporal cross correlation
                            tsi = tiPtr.root.decomps.sources[l,:]
                            tsj = tjPtr.root.decomps.sources[m,:]
                            self.RabDict[(i,j)][l,m] += 0.5*(np.abs((1.0/len(tsi))*np.dot(tsi,tsj)) - tsi.mean()*tsj.mean())
                            # corresponding spatial cross correlation
                            lmatch = self.matchDict[i][l][0]  
                            mmatch = self.matchDict[j][m][0]
                            ssi = siPtr.root.decomps.sources[lmatch,:]
                            ssj = sjPtr.root.decomps.sources[mmatch,:]
                            self.RabDict[(i,j)][l,m] += 0.5*(np.abs((1.0/len(ssi))*np.dot(ssi,ssj)) - ssi.mean()*ssj.mean())
                    tjPtr.close()
                    sjPtr.close()
            siPtr.close()
            tiPtr.close()
        # pickle the result
        rabPtr = open(os.path.join(self.rabDirectory,'rabmatrix.db'),'wb')
        cPickle.dump(self.RabDict,rabPtr,protocol=-1)
        rabPtr.close()
        


    def compute_component_alignments(self):
        '''
        Assembles the alignDict: a dictionary of tuples such that bicar component i will consist of the tuple of
        ICA components in alignDict[i] = (c0,..,cK), along with their spatial matches (from the matchDict); bicar 
        temporal component i will consist of component c0 from ICA run 0, c1 for ICA run 1, . . . , component cK 
        from ICA run K.  The corresponding bicar spatial components will be subsequently assigned via the matching 
        dictionary.
        '''
        # might not have any ica realizations computed
        tICAFiles = sorted(os.listdir(self.temporalDirectory))   
        if len(tICAFiles) == 0:
            raise RAICARICAException
        sICAFiles = sorted(os.listdir(self.spatialDirectory))
        if len(sICAFiles) == 0:
            raise RAICARICAException
        # may not have computed R(a,b); try the version on disk
        if len(self.RabDict) == 0:
            if self.reportLevel > 0:
                print 'No R(a,b) matrix currently in storage; trying version on disk.'
            if not os.path.exists(os.path.join(self.rabDirectory,'rabmatrix.db')):
                raise RAICARRabException
            else:
                rabPtr = open(os.path.join(self.rabDirectory,'rabmatrix.db'),'rb')
                self.RabDict = cPickle.load(rabPtr)
                rabPtr.close()
        if not os.path.exists(self.alnDirectory):
            try:
                os.mkdir(self.alnDirectory)
            except OSError:
                pass
        # need to know how many components to calculate (if any runs exist,
        #    the zeroth one will)
        f0Ptr = tb.openFile(os.path.join(self.temporalDirectory,'tICARun_0.h5'),'r')
        self.nSignals = f0Ptr.root.decomps.sources.shape[0]
        f0Ptr.close()
        for k in range(0,self.nSignals):
            if self.reportLevel > 0:
                print 'Calculating alignment for component %d' % k
            rzIndx,maxElem,compIndx = self.find_max_elem()
            toAlign = self.search_realizations(rzIndx,compIndx)
            self.alignDict[k] = toAlign
            # remove the appropriate rows/cols from Rab so the algorithm can continue
            self.reduce_rab(toAlign)
        # correct the alignment to use actual and not relative indices
        self.correct_alignment()
        # use the matchDict, along with the temporal alignDict, to get the spatial source alignment
        self.spatial_alignment()
        # save the alignment
        fPtr = open(os.path.join(self.alnDirectory,'alignments.db'),'wb')
        cPickle.dump(self.alignDict,fPtr,protocol=-1)
        fPtr.close()
        

    def align_component(self,k):
        '''
        Uses the calculated alignment dictionaries (spatial and temporal) to assemble pairs of a single 
        aligned component, which will be subsequently averaged to make a bicar component.
        '''
        if len(self.alignDict) == 0:
            if self.reportLevel > 0:
                print 'No temporal alignment information currently in storage; trying version on disk.'
            try:
                alnPtr = open(os.path.join(self.alnDirectory,'alignments.db'),'rb')
                self.alignDict = cPickle.load(alnPtr)
                alnPtr.close()
            except:
                raise RAICARAlignmentException
        if len(self.spatialAlignDict) == 0:
            if self.reportLevel > 0:
                print 'No spatial alignment information currently in storage; trying version on disk.'
            try:
                alnPtr = open(os.path.join(self.alnDirectory,'spatial_alignments.db'),'rb')
                self.spatialAlignDict = cPickle.load(alnPtr)
                alnPtr.close()
            except:
                raise RAICARAlignmentException
        if not self.alignDict.has_key(k):
            if self.reportLevel > 0:
                print 'Error.  Requested component %d does not exist.' % k
            return
        # temporal alignment
        tICAFiles = sorted(os.listdir(self.temporalDirectory)) 
        sICAFiles = sorted(os.listdir(self.spatialDirectory))
        if len(tICAFiles) == 0 or len(sICAFiles) == 0:
            raise RAICARICAException
        if self.reportLevel > 0:
            print 'Aligning temporal component %d' % k
        sourcesToAlign = []
        mixColsToAlign = []
        for fi in tICAFiles:
            if self.reportLevel > 1:
                print 'Working on file %s' % fi
            i = np.int(deconstruct_file_name(fi)[1])
            h5Ptr = tb.openFile(os.path.join(self.temporalDirectory,fi),'r')
            sourcesToAlign.append(h5Ptr.root.decomps.sources[self.alignDict[k][i],:])  # source to fetch
            mixColsToAlign.append(h5Ptr.root.decomps.mixing[:,self.alignDict[k][i]]) # mixing element
            h5Ptr.close()
        # temporal source is aligned, form the aligned source and mixing matrix
        alignedSources = np.vstack(sourcesToAlign)
        alignedMixing = np.vstack(mixColsToAlign).T
        fileName = os.path.join(self.alnDirectory,construct_file_name('alnRun_t',k,'h5'))
        h5Ptr = tb.openFile(fileName,mode="w",title='Aligned Component')
        aligned = h5Ptr.createGroup(h5Ptr.root,'aligned','Aligned Component')
        h5Ptr.createArray(aligned,'sources',alignedSources,"S")
        h5Ptr.createArray(aligned,'mixing',alignedMixing,"A")
        h5Ptr.close()
        # repeat for the spatial source
        if self.reportLevel > 0:
            print 'Aligning spatial component %d' % k
        sourcesToAlign = []
        mixColsToAlign = []
        for fi in sICAFiles:
            if self.reportLevel > 1:
                print 'Working on file %s' % fi
            i = np.int(deconstruct_file_name(fi)[1])
            h5Ptr = tb.openFile(os.path.join(self.spatialDirectory,fi),'r')
            sourcesToAlign.append(h5Ptr.root.decomps.sources[self.spatialAlignDict[k][i],:])  # source to fetch
            mixColsToAlign.append(h5Ptr.root.decomps.mixing[:,self.spatialAlignDict[k][i]]) # mixing element
            h5Ptr.close()
        # spatial source is aligned, form the aligned source and mixing matrix
        alignedSources = np.vstack(sourcesToAlign)
        alignedMixing = np.vstack(mixColsToAlign).T
        fileName = os.path.join(self.alnDirectory,construct_file_name('alnRun_s',k,'h5'))
        h5Ptr = tb.openFile(fileName,mode="w",title='Aligned Component')
        aligned = h5Ptr.createGroup(h5Ptr.root,'aligned','Aligned Component')
        h5Ptr.createArray(aligned,'sources',alignedSources,"S")
        h5Ptr.createArray(aligned,'mixing',alignedMixing,"A")
        h5Ptr.close()
        
    
    def construct_bicar_components(self):
        '''
        Averages the aligned ICA runs (both temporal and spatial) and calculates the reproducibility 
        for each component.  avgMethod and canonSigns controls the method of component formation and 
        reproducibility indices calculated.
        '''
        if not os.path.exists(self.racDirectory):
            try:
                os.mkdir(self.racDirectory)
            except OSError:
                pass
        # have to do both temporal and spatial
        tAlnFiles = glob.glob(os.path.join(self.alnDirectory,'alnRun_t_*.h5'))
        sAlnFiles = glob.glob(os.path.join(self.alnDirectory,'alnRun_s_*.h5'))
        if len(tAlnFiles) == 0 or len(sAlnFiles) == 0:
            if self.reportLevel > 0:
                print 'ERROR :  Components have not been aligned yet.'
            return
        # temp variables to hold the answer
        raicarSources = []
        raicarMixing = []
        repro = []
        for f in tAlnFiles:
            if self.reportLevel > 1:
                print 'Constructing temporal bicar component from file %s' % f
            fPtr = tb.openFile(f,'r')
            sc = fPtr.getNode('/aligned/sources').read()
            ac = fPtr.getNode('/aligned/mixing').read()
            fPtr.close()
            if self.canonSigns:
                sc,ac = self.canonicalize_signs(sc,ac)
            methodToUse = self.avgMethod+'_average_aligned_runs'
            avgSource,avgMix,rep = getattr(self,methodToUse)(sc,ac)
            raicarSources.append(avgSource)
            raicarMixing.append(avgMix)
            repro.append(rep)
        # collapse and make a component
        self.temporalSources = np.vstack(raicarSources)
        self.temporalMixing = np.vstack(raicarMixing).T
        self.reproducibility = repro
        # adjust std. dev. of RAICAR sources
        self.temporalSources = standardize(self.temporalSources,stdtype='row')
        # save the result, PyTables again
        h5Ptr = tb.openFile(os.path.join(self.racDirectory,'temporal_components.h5'),mode="w",title='RAICAR Component')
        bicar = h5Ptr.createGroup(h5Ptr.root,'bicar','RAICAR Component')
        h5Ptr.createArray(bicar,'sources',self.temporalSources,"S")
        h5Ptr.createArray(bicar,'mixing',self.temporalMixing,"A")
        h5Ptr.close()
        # repeat the whole thing for the spatial sources
        raicarSources = []
        raicarMixing = []
        repro = []
        for f in sAlnFiles:
            if self.reportLevel > 1:
                print 'Constructing spatial bicar component from file %s' % f
            fPtr = tb.openFile(f,'r')
            sc = fPtr.getNode('/aligned/sources').read()
            ac = fPtr.getNode('/aligned/mixing').read()
            fPtr.close()
            if self.canonSigns:
                sc,ac = self.canonicalize_signs(sc,ac)
            methodToUse = self.avgMethod+'_average_aligned_runs'
            avgSource,avgMix,rep = getattr(self,methodToUse)(sc,ac)
            raicarSources.append(avgSource)
            raicarMixing.append(avgMix)
            repro.append(rep)
        # collapse and make a component
        self.spatialSources = np.vstack(raicarSources)
        self.spatialMixing = np.vstack(raicarMixing).T
        # average reproducibility
        for i in range(0,len(self.reproducibility)):
            self.reproducibility[i] = 0.5*self.reproducibility[i] + 0.5*repro[i]
        # save the result, PyTables again
        h5Ptr = tb.openFile(os.path.join(self.racDirectory,'spatial_components.h5'),mode="w",title='RAICAR Component')
        bicar = h5Ptr.createGroup(h5Ptr.root,'bicar','RAICAR Component')
        h5Ptr.createArray(bicar,'sources',self.spatialSources,"S")
        h5Ptr.createArray(bicar,'mixing',self.spatialMixing,"A")
        h5Ptr.close()
        # this can just be pickled - it's not that large
        fPtr = open(os.path.join(self.racDirectory,'reproducibility.db'),'wb')
        cPickle.dump(self.reproducibility,fPtr,protocol=-1)
        fPtr.close()
    
    
    def compute_raicar_distribution(self):
        '''
        Calcluates the RAICAR distribution (histogram of rz-rz absolute cross correlation coefficients).  
        This can be used to set a significance bound on the BICAR component reproducibility.  If this 
        distribution already exists, the pickled value is read and used to perform the calculations.
        '''
        raicarDist = list()
        if not os.path.exists(self.bkgDirectory):
            try:
                os.mkdir(self.bkgDirectory)
            except OSError:
                pass
        icaFiles = sorted(os.listdir(self.spatialDirectory))
        if len(icaFiles) == 0:
            raise BICARICAException
        for fi in icaFiles:
            fiPtr = tb.openFile(os.path.join(self.spatialDirectory,fi),'r')
            si = fiPtr.getNode('/decomps/sources').read()
            fiPtr.close()
            i = np.int(deconstruct_file_name(fi)[1])
            for fj in icaFiles:
                j = np.int(deconstruct_file_name(fj)[1])
                if j > i:
                    # sources assumed to have unit std. dev. but nonzero mean - will behave badly if not!
                    fjPtr = tb.openFile(os.path.join(self.spatialDirectory,fj),'r')
                    sj = fjPtr.getNode('/decomps/sources').read()
                    fjPtr.close()
                    # break up a complex line
                    siMean = np.reshape(si.mean(axis=1),(si.shape[0],1))
                    sjMean = np.reshape(sj.mean(axis=1),(sj.shape[0],1))
                    flatArray = np.abs((1.0/si.shape[1])*np.dot(si,sj.T) - np.dot(siMean,sjMean.T)).flatten()
                    raicarDist.append(flatArray)
        raicarDist = np.hstack(raicarDist)
        # pickle the result
        rPtr = open(os.path.join(self.bkgDirectory,'raicardist.db'),'wb')
        cPickle.dump(raicarDist,rPtr,protocol=-1)
        rPtr.close()
        
    
    def compute_background(self,method="normal",w=0.5,Rstar=1.0,nSamples=1000):
        '''
        Uses the computed RAICAR distribution (dist. of absolute rz-rz cross-correlation coefficients) for the
        spatial data to estimate a significance bound for the reproducibility, using the method desired.
        INPUT:
            method: string, optional
                "normal" : uses the normal approximation to the bound
                "exact"  : exact draws (nSamples of K*(K-1)/2 values) from the RAICAR distribution
            w : float, optional
                temporal weighting used for reproducibility calculation
            Rstar : float, optional
                assumed maximum reproducibility in temporal dataset
        The bound itself is returned.
        '''
        # create the directory if it does not exist
        if not os.path.exists(self.bkgDirectory):
            try:
                os.mkdir(self.bkgDirectory)
            except OSError:
                pass
        # read the bicar distribution
        rPtr = open(os.path.join(self.bkgDirectory,'raicardist.db'),'rb')
        raicarDist = cPickle.load(rPtr)
        if method == "normal":
            rijbar = np.mean(raicarDist)
            sigmarij = np.std(raicarDist)
            Rc = w*Rstar + (1.0-w)*rijbar + np.sqrt(2/(self.K*(self.K-1)))*2*(1.0-w)*np.sqrt(sigmarij)
            return Rc
        else:
            return 0.0,0.0
            
    
    def read_bicar_components(self,cType='temporal'):
        '''
        Basically just wraps the PyTables bits to load precomputed BICAR components.  They should
        exist in the 'components.h5' file in the /rac directory of the project.
        '''
        if not os.path.exists(self.racDirectory):
            raise RAICARDirectoryExistException(self.racDirectory)
        elif not os.path.exists(os.path.join(self.racDirectory,cType+'_components.h5')):
            raise RAICARComponentException
        # file exists and presumably has something in it
        compFileName = os.path.join(self.racDirectory,cType+'_components.h5')
        h5Ptr = tb.openFile(compFileName,mode="r")
        sources = h5Ptr.getNode('/bicar/sources').read()
        mixing = h5Ptr.getNode('/bicar/mixing').read()
        h5Ptr.close()
        return sources,mixing

    
    def runall(self,X,YT,transfer,similarity='pweighted',degenerate=False):
        '''
        Wrapper to run BICAR from start to finish.  Does not compute the reproducibility
        cutoff - only the BICAR components and their reproducibility.
        '''
        self.kica(X,icaType='temporal')
        self.kica(YT,icaType='spatial')
        self.match_sources(transfer=transfer,similarity=similarity,degenerate=degenerate)
        self.compute_rab()
        self.compute_component_alignments()
        for k in xrange(0,self.nSignals):
            self.align_component(k)
        self.construct_bicar_components()
        return
