'''
Provides basic logging functionality for RAICAR/BICAR.

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

from time import ctime
import re,os
from functools import wraps

def clean_string(inputString):
    parts = re.search("'(\w+\.\w+_'",inputString)
    return parts.group(1)

def create_footer(lkwargs):
    footer = ""
    for kw in lkwargs:
        footer += "\t%s = %s\n"%(kw,lkwargs[kw])
    return footer

class Logger(object):
    '''
    Simple logger to auto-record algorithm options.
    '''
    @classmethod
    def __init__(cls,logfile):
        cls.logfile = logfile
        fileObject = open(cls.logfile,'a')
        timeStamp = "\n[%s] Initializing Log\n"%(ctime())
        fileObject.write(timeStamp)
        fileObject.close()
        # set last event
        cls.lastEvent = timeStamp

    @classmethod
    def log_function_call(cls,event):
        '''
        Main function to log function calls.  Takes one event and multiple args/kwargs.
        '''
        def wrap(f):
            @wraps(f)
            def decorator(*args,**kwargs):
                if hasattr(cls,'logfile'):
                    if cls.lastEvent == event:
                        # just called this function; don't log
                        pass
                    else:
                        cls.lastEvent = event
                        fileObject = open(cls.logfile,'a')
                        header = "[%s]  %s\n"%(ctime(),event)
                        fileObject.write(header)
                        if len(kwargs) > 0:
                            footer = "%s"%(create_footer(kwargs))
                            fileObject.write(footer)
                        fileObject.close()
                else:
                    pass
                return f(*args,**kwargs)
            return decorator
        return wrap


class LogIOException(IOError):
    def __init__(self):
        print "There is a problem with your log file.  Check the path and file name."


