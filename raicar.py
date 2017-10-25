'''
Created on Mar 3, 2011

Implementation of the RAICAR algorithm, using PyTables (for ICA realizations) and cPickle (smaller objects)
in order to avoid memory problems.

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

import tables as tb
import numpy as np
import glob,unittest,cPickle,sys,os,gc

from utilities import standardize, corrmatrix, construct_file_name, deconstruct_file_name
from runlogger import Logger

from scipy import corrcoef,histogram

# decorator for logging
log_function_call = Logger.log_function_call

"""Exceptions for the RAICAR class."""
class RAICARICAException(Exception):
    def __init__(self):
        print "No ICA realizations have been computed.  Run kica() first."

class RAICARRabException(Exception):
    def __init__(self):
        print "R(a,b) matrices have not been computed.  Run compute_rab() first."

class RAICARAlignmentException(Exception):
    def __init__(self):
        print "No alignment information has been determined.  Run compute_component_alignments() first."

class RAICARComponentException(Exception):
    def __init__(self):
        print "No RAICAR components have been calculated yet."

class RAICARDirectoryIOException(Exception):
    def __init__(self):
        print "There is a problem with your input project directory.  Check the path name."

class RAICARProjectCleanException(Exception):
    def __init__(self,dir):
        print "Nothing to clean: directory %s does not exist" % dir

class RAICARDirectoryExistException(Exception):
    def __init__(self,dir):
        print "Project subdirectory %s does not exist" % dir


class RAICAR(object):
    '''
    This is a python implementation of RAICAR (Ranking and Averaging Independent Components by Reproducibility).
    I have modified the originally published algorithm in the following way:

        -I define the reproducibility index as the sum over *all* correlation coefficients (absolute value),
        rather than using a sum of only those absolute correlations larger than a certain value.

        -I make a weighted average of the realization components, using the average absolute correlation.

        -I attempt to canonicalize the signs of the grouped components; I find with many applications of ICA
        the signs of components will be all over the place; I attempt to fix the signs to agree with the
        (arbitrary) sign of the first component added to each pile.

    To run the published version of RAICAR, use the following constructor arguments:
        avgMethod = 'selective', canonSigns = False

    To run the fully updated version, use these:
        avgMethod = 'weighted', canonSigns = True

    References :

        "Ranking and Averaging Independent Component Analysis by Reproducibility (RAICAR)"
        Z. Yang, S. LaConte, X. Weng, and X. Hu, Human Brain Mapping 29:711-725 (2008).

        "BICAR : A New Algorithm for Multiresolution Spatiotemporal Data Fusion"
        K. Brown, S. Grafton, J. Carlson, PLoS ONE 7: e50268 (2012).

    '''
    @log_function_call('Initializing')
    def __init__(self,projDirectory,nSignals=None,K=30,avgMethod='weighted',canonSigns=True,icaMethod=None,icaOptions=None):
        '''
        Parameters:
        ------------
            projDirectory : string
                main directory for the project (storage of ICA realizations, etc.).  Will be
                created if it does not already exist.  Location of:
                    ->./ica : ica realizations
                    ->./rab : dictionary of realization cross-correlation matrices, used to
                                align components and compute reproducibility
                    ->./aln : dictionary of component assignments, and aligned components
                                (before averaging)
                    ->./rac : final set of averaged components and their reproducibility indices

            nSignals : int, optional
                number of signals to extract; default is a full-rank decomposition

            K : int, optional
                number of ICA realizations

            avgMethod : string, optional
                'selective' or 'weighted' : which type of component averaging to use

            canonSigns : bool, optional
                perform sign canonicalization?


            icaMethod : function, optional
                a function to use for fastICA decompositions. if icaMethod=None, RAICAR will try to use
                fastica from the pyica package.  If you supply your own function, it should follow the
                return convention described in the README file.  The method calling sequence can be
                f(X,nSources,...), where X is the data matrix,nSources is the requested number of sources
                , and then any number of keyword arguments follow; icaOptions specifies necessary keyword
                arguments


            icaOptions : dict, optional
                arguments to pass to fastICA; defaults correspond to kwargs in pyica.fastica
                    -algorithm: string
                    -decorrelation : string
                    -nonlinearity : string
                    -alpha : float in [0,1]
                    -maxIterations : integer
                    -tolerance : float
            '''
        if not os.path.exists(projDirectory):
            os.mkdir(projDirectory)
        # create a new project
        self.projDirectory = projDirectory
        self.icaDirectory = os.path.join(self.projDirectory,'ica')
        self.rabDirectory = os.path.join(self.projDirectory,'rab')
        self.alnDirectory = os.path.join(self.projDirectory,'aln')
        self.racDirectory = os.path.join(self.projDirectory,'rac')
        self.K = K
        self.avgMethod = avgMethod
        self.canonSigns = canonSigns
        self.nSignals = nSignals
        # tuple-indexed list of K*(K-1)/2 cross-correlation matrices from K realizations of ICA
        self.RabDict = dict()
        # alignment information for ICS;
        #    alignDict[j] = [c0,c1,...,cK-1]
        self.alignDict = {}
        # will hold the eventual raicar sources/mixing matrices
        self.raicarSources = None
        self.raicarMixing = None
        if icaOptions is None:
            self.icaOptions = dict()
            self.icaOptions['algorithm'] = 'parallel fp'
            self.icaOptions['decorrelation'] = 'mdum'
            self.icaOptions['nonlinearity'] = 'logcosh'
            self.icaOptions['alpha'] = 1.0
            self.icaOptions['maxIterations'] = 500
            self.icaOptions['tolerance'] = 1.0e-05
        else:
            self.icaOptions = icaOptions
        if icaMethod is None:
            from pyica import fastica
            self.ica = fastica
        else:
            self.ica = icaMethod


    def find_max_elem(self):
        '''
        Searches the list of realization-realization cross correlation matrices to find the single largest element;
        the realization indices (a,b), value (maxElem), and component indices (m,n) are returned as a tuple.
        '''
        # ((a,b),value,(m,n) for every matrix) max value in each matrix
        matrixMax = [(k,self.RabDict[k].max(),np.unravel_index(self.RabDict[k].argmax(),self.RabDict[k].shape)) for k in self.RabDict]
        # maximum of the maxima
        bigIndx = np.argmax(zip(*matrixMax)[1])
        return matrixMax[bigIndx]


    def search_realizations(self,rzIndx,compIndx):
        '''
        Accepts an (a,b) tuple of realizations (rzIndx) and an (m,n) tuple of component indices (compIndx) and
        searches all the other K-2 realizations for components to match with C_am and C_bn.  Each run of
        search_realizations() returns a tuple of ints toAlign = (c0,c1,...,cK-1) which gives
        components to group - c0 goes with realization 0, etc.
        '''
        a,b = rzIndx
        m,n = compIndx
        rzToSearch = [x for x in xrange(0,self.K) if x is not a and x is not b]
        toAlign = [(a,m),(b,n)]
        for r in rzToSearch:
            rzOne = [a,r]
            rzTwo = [r,b]
            # row/col search is tricky because of RabDict storage
            if rzOne[0] > rzOne[1]:
                # search the transpose
                x = (rzOne[1],rzOne[0])
                Mrow = self.RabDict[x].T
            else:
                Mrow = self.RabDict[tuple(rzOne)]
            if rzTwo[0] > rzTwo[1]:
                # search the transpose
                x = (rzTwo[1],rzTwo[0])
                Mcol = self.RabDict[x].T
            else:
                Mcol = self.RabDict[tuple(rzTwo)]
            (rowMax,pi) = (Mrow[m,:].max(),Mrow[m,:].argmax())
            (colMax,qi) = (Mcol[:,n].max(),Mcol[:,n].argmax())
            # this will do the right thing whether pi = qi or not
            whichComp = pi if rowMax > colMax else qi
            toAlign.append((r,whichComp))
        toAlign.sort()
        return list(zip(*toAlign)[1])


    def reduce_rab(self,toAlign):
        '''
        Deletes a single row and column from each of the Rab matrices.
        '''
        for r1 in range(0,self.K):
            for r2 in range(r1+1,self.K):
                rz = (r1,r2)
                rcDel = (toAlign[r1],toAlign[r2])
                self.RabDict[rz] = np.delete(np.delete(self.RabDict[rz],rcDel[0],0),rcDel[1],1)


    def correct_alignment(self):
        '''
        During the course of the algorithm, row/col deletion in R(a,b) causes alignment indices to be relative
        and not absolute - for example, the last component has indices [0,0,...,0], since at that step only
        one element is left in each realization matrix.  This function restores actual index numbers to the
        alignment.
        '''
        if len(self.alignDict) == 0:
            raise RAICARAlignmentException
        newIndexSet = []
        # using alignDict.values() may or may not be safe.
        for k in range(0,self.K):
            newIndexSet.append(self.index_remap((np.asarray(self.alignDict.values())[:,k]).tolist()))
        for k in self.alignDict:
            self.alignDict[k] = ((np.asarray(newIndexSet).T)[k,:]).tolist()


    def index_remap(self,indToRemap):
        '''
        Remaps one set of indices from a complete decimation from relative to absolute indices - necessary
        for nondestructively constructing the raicar alignments.
        '''
        remappedIndex = [0 for k in indToRemap]
        i_r = dict()
        for k in range(0,len(indToRemap)-1):
            i_r[k] = {}.fromkeys(range(0,len(indToRemap)-k-1))
        for oKey in i_r.keys():
            for iKey in i_r[oKey].keys():
                i_r[oKey][iKey] = iKey if iKey < indToRemap[oKey] else iKey+1
        for posToRemap in range(0,len(indToRemap)):
            temp_remap = indToRemap[posToRemap]
            for k in range(posToRemap-1,-1,-1):
                temp_remap = i_r[k][temp_remap]
            remappedIndex[posToRemap] = temp_remap
        return remappedIndex

    @log_function_call('Canonicalizing signs')
    def canonicalize_signs(self,sources,mixing):
        '''
        Accepts an set of sources and corresponding mixing matrices from an ICA component (should be a realization component,
        as this operation makes no sense for regular ICA realizations) and fixes the signs of the realizations, using the sign
        of the inter-source cross correlations.  Specifically, the 0th source is arbitrarily deemed to have the canonical sign;
        components which correlate positively with this component keep the same signs, and those which negatively correlate
        have their signs reversed.  This WILL NOT ensure all source-source correlations are positive, but will tend to cause
        the 'well matched' components to have the same sign.
        '''
        compSigns = np.sign(corrcoef(sources)[0,:])
        for i in range(1,sources.shape[0]):
            sources[i,:] = compSigns[i]*sources[i,:]
            mixing[:,i] = compSigns[i]*mixing[:,i]
        return sources,mixing

    @log_function_call('Selectively averaging aligned components')
    def selective_average_aligned_runs(self,sources,mixing):
        '''
        Averages one aligned ICA run and calculates a reproducibility index.  This version uses the original
        definition in Yang et al.
        '''
        # threshold for inclusion
        thresh = 0.7
        corrsToSum = np.triu(np.abs(corrcoef(sources)),1).flatten()
        rep = (corrsToSum[np.nonzero(corrsToSum > thresh)].sum())/(0.5*self.K*(self.K-1))
        # now only add a component to the average if there is at least one correlation with the other RCs > threshold
        #    the > 1 statement is because the diagonal elements are always 1.0, so there will always be at least one
        #    cross-correlation (namely self-correlation) which is bigger than 1
        toInclude = ((np.abs(corrcoef(sources)) > thresh).sum(axis=0) > 1)
        return sources[toInclude,:].mean(axis=0),mixing[:,toInclude].mean(axis=1),rep

    @log_function_call('Weighted averaging aligned components')
    def weighted_average_aligned_runs(self,sources,mixing):
        '''
        Averages one aligned ICA run and calculates the reproducibility for each component.  This version does not
        add only super-threshold CCs to the reproducibililty index, and it uses a weighted average to form the
        average components.  The weights are defined as w_i = sum_{j neq i} SCC(i,j).
        '''
        rep = np.triu(np.abs(corrcoef(sources)),1).sum()/(0.5*self.K*(self.K-1))
        rWeights = np.asarray([(np.abs(corrcoef(sources)[j,:]).sum() - 1.0)/(sources.shape[0]-1) for j in range(0,sources.shape[0])])[:,np.newaxis]
        return ((rWeights*sources).sum(axis=0))/(rWeights.sum()),((mixing*rWeights.T).sum(axis=1))/(rWeights.sum()),rep

    @log_function_call('Cleaning project')
    def clean_project(self):
        '''
        Removes all files in the subdirectories of the project directory, as well as the directories.
        Subdirectories which do not exist (having not yet been created) are skipped.
        '''
        projDirectories = [self.icaDirectory,self.rabDirectory,self.alnDirectory,self.racDirectory]
        for d in projDirectories:
            if not os.path.exists(d):
                print "Nothing to clean: directory %s does not exist" % d
            else:
                print "Cleaning %s" % d
                files = os.listdir(d)
                for f in files:
                    os.remove(os.path.join(d,f))

    @log_function_call('Running K-fold ICA')
    def kica(self,X):
        '''
        Runs K realizations of ICA (method dictated by constructor argument icaMethod), decomposing data matrix X
        into A*S, for sources S and mixing matrix A.  Resulting realizations are stored in a PyTable in
        the /ica directory.
        '''
        if not os.path.exists(self.icaDirectory):
            try:
                os.mkdir(self.icaDirectory)
            except OSError:
                pass
        gc.collect()
        # files to construct
        icaToMake = [os.path.join(self.icaDirectory,construct_file_name('icaRun',x,'h5')) for x in range(0,self.K)]
        if self.nSignals is None:
            self.nSignals = X.shape[0] # full decomp
        for icaFile in icaToMake:
            if not os.path.exists(icaFile):
                print 'Running ICA realization %s' % icaFile
                A,W,S = self.ica(X,nSources=self.nSignals,**self.icaOptions)
                # write the results to a PyTable
                h5Ptr = tb.open_file(icaFile,mode="w",title='ICA Realization')
                decomps = h5Ptr.create_group(h5Ptr.root,'decomps','ICA Decompositions')
                h5Ptr.create_array(decomps,'sources',S,"S")
                h5Ptr.create_array(decomps,'mixing',A,"A")
                h5Ptr.close()
            else:
                print 'ICA realization %s already exists.  Skipping.' % icaFile

    @log_function_call('Computing R(a,b) matrices')
    def compute_rab(self):
        '''
        Uses the current set of ICA realizations (pytabled) to compute K*(K-1)/2 cross-correlation matrices;
        they are indexed via tuples.  R(a,b) is much smaller than the ICA realizations (all R(a,b) matrices
        are generally smaller than ONE realization), so R(a,b) is also retained in memory. Recomputation of
        the R(a,b) matrices is forced.
        '''
        if not os.path.exists(self.rabDirectory):
            try:
                os.mkdir(self.rabDirectory)
            except OSError:
                pass
        icaFiles = sorted(os.listdir(self.icaDirectory))
        if len(icaFiles) == 0:
            raise RAICARICAException
        for fi in icaFiles:
            fiPtr = tb.open_file(os.path.join(self.icaDirectory,fi),'r')
            si = fiPtr.get_node('/decomps/sources').read()
            fiPtr.close()
            i = np.int(deconstruct_file_name(fi)[1])
            print 'Working on R(%d,b)'%i
            for fj in icaFiles:
                j = np.int(deconstruct_file_name(fj)[1])
                if j > i:
                    # sources assumed to have unit std. dev. but nonzero mean - will behave badly if not!
                    fjPtr = tb.open_file(os.path.join(self.icaDirectory,fj),'r')
                    sj = fjPtr.get_node('/decomps/sources').read()
                    fjPtr.close()
                    self.RabDict[(i,j)] = np.abs(corrmatrix(si,sj))
        # pickle the result
        rabPtr = open(os.path.join(self.rabDirectory,'rabmatrix.db'),'wb')
        cPickle.dump(self.RabDict,rabPtr,protocol=-1)
        rabPtr.close()


    def compute_scc_histogram(self):
        '''
        Computes a histogram of abs correlation coefficients from the pickled R(a,b) matrix.
        '''
        if not os.path.exists(self.rabDirectory):
            raise RAICARRabException
        if not os.path.exists(os.path.join(self.rabDirectory,'rabmatrix.db')):
            raise RAICARRabException
        rabPtr = open(os.path.join(self.rabDirectory,'rabmatrix.db'),'rb')
        RabDict = cPickle.load(rabPtr)
        rabPtr.close()
        rPDF = dict.fromkeys(['bin edges','counts','bar width'])
        rPDF['bin edges'] = np.linspace(0,1.0,101)
        rPDF['counts'],_ = histogram(a=np.hstack(RabDict.values()[i].flatten() for i in range(0,len(RabDict))),bins=rPDF['bin edges'])
        rPDF['bin edges'] = rPDF['bin edges'][0:-1]
        rPDF['bar width'] = 0.01
        return rPDF

    @log_function_call('Computing component alignments')
    def compute_component_alignments(self):
        '''
        Assembles the alignDict: a dictionary of tuples such that raicar component i will consist of the tuple of
        ica components in alignDict[i] = (c0,..,cK); raicar component i will consist of component c0 from ica run
        0, c1 for ica run 1, . . . , component cK from ica run K.
        '''
        # might not have any ica realizations computed
        icaFiles = sorted(os.listdir(self.icaDirectory))
        if len(icaFiles) == 0:
            raise RAICARICAException
        # may not have computed R(a,b); try the version on disk
        if len(self.RabDict) == 0:
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
        gc.collect()
        # need to know how many components to calculate (if any runs exist,
        #    the zeroth one will)
        f0Ptr = tb.open_file(os.path.join(self.icaDirectory,'icaRun_0.h5'),'r')
        s0 = f0Ptr.get_node('/decomps/sources').read()
        f0Ptr.close()
        nComp = s0.shape[0]
        del s0
        for k in range(0,nComp):
            print 'Calculating alignment for component %d' % k
            rzIndx,maxElem,compIndx = self.find_max_elem()
            toAlign = self.search_realizations(rzIndx,compIndx)
            self.alignDict[k] = toAlign
            # remove the appropriate rows/cols from Rab so the algorithm can continue
            self.reduce_rab(toAlign)
        # correct the alignment to use actual and not relative indices
        self.correct_alignment()
        # save the alignment
        fPtr = open(os.path.join(self.alnDirectory,'alignments.db'),'wb')
        cPickle.dump(self.alignDict,fPtr,protocol=-1)
        fPtr.close()

    @log_function_call('Aligning component')
    def align_component(self,k):
        '''
        Uses the alignment dictionary to assemble a single aligned component, which will be subsequently
        averaged to make a raicar component.
        '''
        gc.collect()
        if not os.path.exists(self.alnDirectory):
            try:
                os.mkdir(self.alnDirectory)
            except OSError:
                pass
        if len(self.alignDict) == 0:
            print 'No alignment information currently in storage; trying version on disk.'
            if not os.path.exists(os.path.join(self.alnDirectory,'alignments.db')):
                raise RAICARAlignmentException
            else:
                alnPtr = open(os.path.join(self.alnDirectory,'alignments.db'),'rb')
                self.alignDict = cPickle.load(alnPtr)
                alnPtr.close()
        if not self.alignDict.has_key(k):
            print 'Error.  Requested component %d does not exist.' % k
            return
        icaFiles = sorted(os.listdir(self.icaDirectory))
        if len(icaFiles) == 0:
            raise RAICARICAException
        print 'Aligning component %d' % k
        sourcesToAlign = []
        mixColsToAlign = []
        for fi in icaFiles:
            print 'Working on file %s' % fi
            i = np.int(deconstruct_file_name(fi)[1])
            h5Ptr = tb.open_file(os.path.join(self.icaDirectory,fi),'r')
            sourcesToAlign.append(h5Ptr.root.decomps.sources[self.alignDict[k][i],:])  # source to fetch
            mixColsToAlign.append(h5Ptr.root.decomps.mixing[:,self.alignDict[k][i]]) # mixing element
            h5Ptr.close()
        # source is aligned, form the aligned source and mixing matrix
        alignedSources = np.vstack(sourcesToAlign)
        alignedMixing = np.vstack(mixColsToAlign).T
        fileName = os.path.join(self.alnDirectory,construct_file_name('alnRun',k,'h5'))
        h5Ptr = tb.open_file(fileName,mode="w",title='Aligned Component')
        aligned = h5Ptr.create_group(h5Ptr.root,'aligned','Aligned Component')
        h5Ptr.create_array(aligned,'sources',alignedSources,"S")
        h5Ptr.create_array(aligned,'mixing',alignedMixing,"A")
        h5Ptr.close()

    @log_function_call('Constructing RAICAR components')
    def construct_raicar_components(self):
        '''
        Averages the aligned ICA runs and calculates the reproducibility for each component.  avgMethod and
        canonSigns controls the method of component formation and reproducibility indices calculated.
        '''
        if not os.path.exists(self.racDirectory):
            try:
                os.mkdir(self.racDirectory)
            except OSError:
                pass
        alnFiles = glob.glob(os.path.join(self.alnDirectory,'alnRun_*.h5'))
        if len(alnFiles) == 0:
            print 'ERROR :  Components have not been aligned yet.'
            return
        # temp variables to hold the answer
        gc.collect()
        raicarSources = []
        raicarMixing = []
        repro = []
        for f in alnFiles:
            print 'Constructing raicar comfponent from file %s' % f
            fPtr = tb.open_file(f,'r')
            sc = fPtr.get_node('/aligned/sources').read()
            ac = fPtr.get_node('/aligned/mixing').read()
            fPtr.close()
            if self.canonSigns:
                sc,ac = self.canonicalize_signs(sc,ac)
            methodToUse = self.avgMethod+'_average_aligned_runs'
            avgSource,avgMix,rep = getattr(self,methodToUse)(sc,ac)
            raicarSources.append(avgSource)
            raicarMixing.append(avgMix)
            repro.append(rep)
        # collapse and make a component
        self.raicarSources = np.vstack(raicarSources)
        self.raicarMixing = np.vstack(raicarMixing).T
        self.reproducibility = repro
        # adjust std. dev. of RAICAR sources
        self.raicarSources = standardize(self.raicarSources,stdtype='row')
        # save the result, PyTables again
        h5Ptr = tb.open_file(os.path.join(self.racDirectory,'components.h5'),mode="w",title='RAICAR Component')
        raicar = h5Ptr.create_group(h5Ptr.root,'raicar','RAICAR Component')
        h5Ptr.create_array(raicar,'sources',self.raicarSources,"S")
        h5Ptr.create_array(raicar,'mixing',self.raicarMixing,"A")
        h5Ptr.close()
        # this can just be pickled - it's not that large
        fPtr = open(os.path.join(self.racDirectory,'reproducibility.db'),'wb')
        cPickle.dump(self.reproducibility,fPtr,protocol=-1)
        fPtr.close()


    def read_raicar_components(self):
        '''
        Basically just wraps the PyTables bits to load precomputed RAICAR components.  They should
        exist in the 'components.h5' file in the /rac directory of the project.
        '''
        if not os.path.exists(self.racDirectory):
            raise RAICARDirectoryExistException(self.racDirectory)
        elif not os.path.exists(os.path.join(self.racDirectory,'components.h5')):
            raise RAICARComponentException
        # file exists and presumably has something in it
        compFileName = os.path.join(self.racDirectory,'components.h5')
        h5Ptr = tb.open_file(compFileName,mode="r")
        sources = h5Ptr.get_node('/raicar/sources').read()
        mixing = h5Ptr.get_node('/raicar/mixing').read()
        h5Ptr.close()
        return sources,mixing

    def read_reproducibility(self):
        '''
        Wraps the unpickling of the reproduciblity into a function; checks that the reproducibility
        file exists first.
        '''
        if not os.path.exists(self.racDirectory):
            raise RAICARDirectoryExistException(self.racDirectory)
        elif not os.path.exists(os.path.join(self.racDirectory,'reproducibility.db')):
            raise RAICARComponentException
        fPtr = open(os.path.join(self.racDirectory,'reproducibility.db'),'rb')
        repro = cPickle.load(fPtr)
        fPtr.close()
        return repro

    def runall(self,X):
        '''
        Runs the entire RAICAR pipeline, from ICA to RAICAR source construction and reproducibility
        calculation.
        '''
        self.kica(X)
        self.compute_rab()
        self.compute_component_alignments()
        for k in xrange(self.nSignals):
            self.align_component(k)
        self.construct_raicar_components()
        return
