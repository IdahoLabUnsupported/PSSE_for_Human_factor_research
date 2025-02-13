# Copyright 2025, Battelle Energy Alliance, LLC All Rights Reserved
# ====================================================================================================
'''
This is an example file showing how to use "subsystem data retrieval APIs (API Manual, Chapter 8")
from Python to generate power flow results report.
    Input : Solved PSS(R)E saved case file name
    Output: Report file name to save
    When 'savfile' is provided, FNSL with default options is used to solve the case.
    When 'savfile' is not provided, it uses solved Case from PSS(R)E memory.
    When 'rptfile' is provided, report is saved in ASCII text file 'rptfile'.
    When 'rptfile' is not provided, it produces report in PSS(R)E report window.

The subsystem data retrieval APIs return values as List of Lists. For example:
When "abusint" API is called with "istrings" as defined below:
 idata=[[list of 'number'],[list of 'type'],[],[],[],[list of 'dummy']]

This example is written such that, such returned lists are converted into dictionary with
keys as strings specified in "istrings". This makes it easier to refer and use these lists.
    ibuses = array2dict(istrings, idata)

So ibuses['number'] gives the bus numbers returned by "abusint"   istrings = ['number','type','area','zone','owner','dummy']
    ierr, idata = psspy.abusint(sid, flag_bus, istrings)
The returned list will have format:
    .

---------------------------------------------------------------------------------
How to use this file?

As showed in __main__ (end of this file)
- Enable PSSE version specific environment, as an example:
    import psse35

- call funtion
    pout_report(savfile, outpath)
    or
    pout_report()  <-- savnw.sav and savnw.dfx must exist in working folder.

'''
# ----------------------------------------------------------------------------------------------------
import sys, os
import numpy as np
import time

# ----------------------------------------------------------------------------------------------------
# Function to convert arrays to a dictionary with the specified keys.
def array2dict(dict_keys, dict_values):
    '''Convert array to dictionary of arrays.
    Returns dictionary as {dict_keys:dict_values}
    '''
    tmpdict = {}
    for i in range(len(dict_keys)):
        tmpdict[dict_keys[i].lower()] = dict_values[i]
    return tmpdict

# ----------------------------------------------------------------------------------------------------
def busindexes(busnum, busnumlist):
    '''Find indexes of a bus in list of buses.
    Returns list with indexes of 'busnum' in 'busnumlist'.
    '''
    busidxes = []
    startidx = 0
    buscounts = busnumlist.count(busnum)
    if buscounts:
        for i in range(buscounts):
            tmpidx = busnumlist.index(busnum,startidx)
            busidxes.append(tmpidx)
            startidx = tmpidx+1
    return busidxes

# ----------------------------------------------------------------------------------------------------
def splitstring_commaspace(tmpstr):
    '''Split string first at comma and then by space. Example:
    Input  tmpstr = a1       a2,  ,a4 a5 ,,,a8,a9
    Output strlst = ['a1', 'a2', ' ', 'a4', 'a5', ' ', ' ', 'a8', 'a9']
    '''
    strlst = []
    commalst = tmpstr.split(',')
    for each in commalst:
        eachlst = each.split()
        if eachlst:
            strlst.extend(eachlst)
        else:
            strlst.extend(' ')

    return strlst

# -----------------------------------------------------------------------------------------------------

def get_output_dir(outpath):
    # if called from PSSE's Example Folder, create report in subfolder 'Output_Pyscript'

    if outpath:
        outdir = outpath
        if not os.path.exists(outdir): os.mkdir(outdir)
    else:
        outdir = os.getcwd()
        cwd = outdir.lower()
        i = cwd.find('pti')
        j = cwd.find('psse')
        k = cwd.find('example')
        if i>0 and j>i and k>j:     # called from Example folder
            outdir = os.path.join(outdir, 'Output_Pyscript')
            if not os.path.exists(outdir): os.mkdir(outdir)

    return outdir

# -----------------------------------------------------------------------------------------------------

def get_output_filename(outpath, fnam):

    p, nx = os.path.split(fnam)
    if p:
        retvfile = fnam
    else:
        outdir = get_output_dir(outpath)
        retvfile = os.path.join(outdir, fnam)

    return retvfile 

# ----------------------------------------------------------------------------------------------------
def pout_report(savfile='savnw.sav', outpath=None,AreaID=1):
    '''Generates power flow result report.
    When 'savfile' is provided, FNSL with default options is used to solve the case.
    When 'savfile' is not provided, it uses solved Case from PSS(R)E memory.
    When 'rptfile' is provided, report is saved in ASCII text file 'rptfile'.
    When 'rptfile' is not provided, it produces report in PSS(R)E report window.
    '''

    # import psspy

    # psspy.psseinit()

    # Set Save and Report files according to input file names
    if savfile:
        ierr = psspy.case(savfile)
        if ierr != 0: return
        fpath, fext = os.path.splitext(savfile)
        if not fext: savfile = fpath + '.sav'
        #ierr = psspy.fnsl([0,0,0,1,1,0,0,0])
        #if ierr != 0: return
    else:   # saved case file not provided, check if working case is in memory
        ierr, nbuses = psspy.abuscount(-1,2)
        if ierr == 0:
            savfile, snapfile = psspy.sfiles()
        else:
            print('\n No working case in memory.')
            print(' Either provide a Saved case file name or open Saved case in PSS(R)E.')
            return

    p, nx = os.path.split(savfile)
    n, x  = os.path.splitext(nx)
    rptfile = get_output_filename(outpath, 'pout_'+n+'test.txt')


    # ================================================================================================
    # PART 1: Get the required results data
    # ================================================================================================

    # Select what to report
    # if psspy.bsysisdef(0):
        # sid = 0
    # else:   # Select subsytem with all buses
        # sid = -1

    # flag_bus     = 1    # in-service
    # flag_plant   = 1    # in-service
    # flag_load    = 1    # in-service
    # flag_swsh    = 1    # in-service
    # flag_brflow  = 1    # in-service
    # owner_brflow = 1    # use bus ownership, ignored if sid is -ve
    # ties_brflow  = 5    # ignored if sid is -ve

    # ------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------
    # Case Title
    titleline1, titleline2 = psspy.titldt()

    # ------------------------------------------------------------------------------------------------
    # Bus Data
    # Bus Data - Integer
    istrings = ['number','type','area','zone','owner','dummy']
    ierr, idata = psspy.abusint(sid, flag_bus, istrings)
    if ierr:
        print('(1) psspy.abusint error = %d' % ierr)
        return
    ibuses = array2dict(istrings, idata)
    # Bus Data - Real
    rstrings = ['base','pu','kv','angle','angled','mismatch','o_mismatch']
    ierr, rdata = psspy.abusreal(sid, flag_bus, rstrings)
    if ierr:
        print('(1) psspy.abusreal error = %d' % ierr)
        return
    rbuses = array2dict(rstrings, rdata)
    # Bus Data - Complex
    xstrings = ['voltage','shuntact','o_shuntact','shuntnom','o_shuntnom','mismatch','o_mismatch']
    ierr, xdata = psspy.abuscplx(sid, flag_bus, xstrings)
    if ierr:
        print('(1) psspy.abuscplx error = %d' % ierr)
        return
    xbuses = array2dict(xstrings, xdata)
    # Bus Data - Character
    cstrings = ['name','exname']
    ierr, cdata = psspy.abuschar(sid, flag_bus, cstrings)
    if ierr:
        print('(1) psspy.abuschar error = %d' % ierr)
        return
    cbuses = array2dict(cstrings, cdata)

    # Store bus data for all buses
    ibusesall={};rbusesall={};xbusesall={};cbusesall={};
    if sid == -1:
        ibusesall=ibuses
        rbusesall=rbuses
        xbusesall=xbuses
        cbusesall=cbuses
    else:
        ierr, idata = psspy.abusint(-1, flag_bus, istrings)
        if ierr:
            print('(2) psspy.abusint error = %d' % ierr)
            return
        ibusesall = array2dict(istrings, idata)

        ierr, rdata = psspy.abusreal(-1, flag_bus, rstrings)
        if ierr:
            print('(2) psspy.abusreal error = %d' % ierr)
            return
        rbusesall = array2dict(rstrings, rdata)

        ierr, xdata = psspy.abuscplx(-1, flag_bus, xstrings)
        if ierr:
            print('(2) psspy.abuscplx error = %d' % ierr)
            return
        xbusesall = array2dict(xstrings, xdata)

        ierr, cdata = psspy.abuschar(-1, flag_bus, cstrings)
        if ierr:
            print('(2) psspy.abuschar error = %d' % ierr)
            return
        cbusesall = array2dict(cstrings, cdata)
    
    # ------------------------------------------------------------------------------------------------
    # Plant Bus Data
    # Plant Bus Data - Integer
    istrings = ['number','type','area','zone','owner','dummy', 'status','ireg']
    ierr, idata = psspy.agenbusint(sid, flag_plant, istrings)
    if ierr:
        print('psspy.agenbusint error = %d' % ierr)
        return
    iplants = array2dict(istrings, idata)
    # Plant Bus Data - Real
    rstrings = ['base','pu','kv','angle','angled','iregbase','iregpu','iregkv','vspu','vskv','rmpct',
                'pgen',  'qgen',  'mva', 'percent', 'pmax',  'pmin',  'qmax',  'qmin',  'mismatch',
                'o_pgen','o_qgen','o_mva','o_pmax','o_pmin','o_qmax','o_qmin','o_mismatch']
    ierr, rdata = psspy.agenbusreal(sid, flag_plant, rstrings)
    if ierr:
        print('psspy.agenbusreal error = %d' % ierr)
        return
    rplants = array2dict(rstrings, rdata)
    # Plant Bus Data - Complex
    xstrings = ['voltage','pqgen','mismatch','o_pqgen','o_mismatch']
    ierr, xdata = psspy.agenbuscplx(sid, flag_plant, xstrings)
    if ierr:
        print('psspy.agenbusreal error = %d' % ierr)
        return
    xplants = array2dict(xstrings, xdata)
    # Plant Bus Data - Character
    cstrings = ['name','exname','iregname','iregexname']
    ierr, cdata = psspy.agenbuschar(sid, flag_plant, cstrings)
    if ierr:
        print('psspy.agenbuschar error = %d' % ierr)
        return
    cplants = array2dict(cstrings, cdata)

    # ------------------------------------------------------------------------------------------------
    # Load Data - based on Individual Loads Zone/Area/Owner subsystem
    # Load Data - Integer
    istrings = ['number','area','zone','owner','status']
    ierr, idata = psspy.aloadint(sid, flag_load, istrings)
    if ierr:
        print('psspy.aloadint error = %d' % ierr)
        return
    iloads = array2dict(istrings, idata)
    # Load Data - Real
    rstrings = ['mvaact','mvanom','ilact','ilnom','ylact','ylnom','totalact','totalnom','o_mvaact',
                'o_mvanom','o_ilact','o_ilnom','o_ylact','o_ylnom','o_totalact','o_totalnom']
    ierr, rdata = psspy.aloadreal(sid, flag_load, rstrings)
    if ierr:
        print('psspy.aloadreal error = %d' % ierr)
        return
    rloads = array2dict(rstrings, rdata)
    # Load Data - Complex
    xstrings = rstrings
    ierr, xdata = psspy.aloadcplx(sid, flag_load, xstrings)
    if ierr:
        print('psspy.aloadcplx error = %d' % ierr)
        return
    xloads = array2dict(xstrings, xdata)
    # Load Data - Character
    cstrings = ['id','name','exname']
    ierr, cdata = psspy.aloadchar(sid, flag_load, cstrings)
    if ierr:
        print('psspy.aloadchar error = %d' % ierr)
        return
    cloads = array2dict(cstrings, cdata)

    # ------------------------------------------------------------------------------------------------
    # Total load on a bus
    totalmva={}; totalil={}; totalyl={}; totalys={}; totalysw={}; totalload={}; busmsm={}
    for b in ibuses['number']:
        ierr, ctmva = psspy.busdt2(b,'MVA','ACT')
        if ierr==0: totalmva[b]=ctmva

        ierr, ctil = psspy.busdt2(b,'IL','ACT')
        if ierr==0: totalil[b]=ctil

        ierr, ctyl = psspy.busdt2(b,'YL','ACT')
        if ierr==0: totalyl[b]=ctyl

        ierr, ctys = psspy.busdt2(b,'YS','ACT')
        if ierr==0: totalys[b]=ctys

        ierr, ctysw = psspy.busdt2(b,'YSW','ACT')
        if ierr==0: totalysw[b]=ctysw

        ierr, ctld = psspy.busdt2(b,'TOTAL','ACT')
        if ierr==0: totalload[b]=ctld

        #Bus mismstch
        ierr, msm = psspy.busmsm(b)
        if ierr != 1: busmsm[b]=msm

    # ------------------------------------------------------------------------------------------------
    # Switched Shunt Data
    # Switched Shunt Data - Integer
    istrings = ['number','type','area','zone','owner','dummy','mode','ireg','blocks',
                'stepsblock1','stepsblock2','stepsblock3','stepsblock4','stepsblock5',
                'stepsblock6','stepsblock7','stepsblock8']
    ierr, idata = psspy.aswshint(sid, flag_swsh, istrings)
    if ierr:
        print('psspy.aswshint error = %d' % ierr)
        return
    iswsh = array2dict(istrings, idata)
    # Switched Shunt Data - Real (Note: Maximum allowed NSTR are 50. So they are split into 2)
    rstrings = ['base','pu','kv','angle','angled','vswhi','vswlo','rmpct','bswnom','bswmax',
                'bswmin','bswact','bstpblock1','bstpblock2','bstpblock3','bstpblock4','bstpblock5',
                'bstpblock6','bstpblock7','bstpblock8','mismatch']
    rstrings1 = ['o_bswnom','o_bswmax','o_bswmin','o_bswact','o_bstpblock1',
                 'o_bstpblock2','o_bstpblock3','o_bstpblock4','o_bstpblock5','o_bstpblock6',
                 'o_bstpblock7','o_bstpblock8','o_mismatch']
    ierr, rdata = psspy.aswshreal(sid, flag_swsh, rstrings)
    if ierr:
        print('(1) psspy.aswshreal error = %d' % ierr)
        return
    rswsh = array2dict(rstrings, rdata)
    ierr, rdata1 = psspy.aswshreal(sid, flag_swsh, rstrings1)
    if ierr:
        print('(2) psspy.aswshreal error = %d' % ierr)
        return
    rswsh1 = array2dict(rstrings1, rdata1)
    for k, v in rswsh1.items():
        rswsh[k]=v
    # Switched Shunt Data - Complex
    xstrings = ['voltage','yswact','mismatch','o_yswact','o_mismatch']
    ierr, xdata = psspy.aswshcplx(sid, flag_swsh, xstrings)
    if ierr:
        print('psspy.aswshcplx error = %d' % ierr)
        return
    xswsh = array2dict(xstrings, xdata)
    # Switched Shunt Data - Character
    cstrings = ['vscname','name','exname','iregname','iregexname']
    ierr, cdata = psspy.aswshchar(sid, flag_swsh, cstrings)
    if ierr:
        print('psspy.aswshchar error = %d' % ierr)
        return
    cswsh = array2dict(cstrings, cdata)

    # ------------------------------------------------------------------------------------------------
    # Branch Flow Data
    # Branch Flow Data - Integer
    istrings = ['fromnumber','tonumber','status','nmeternumber','owners','own1','own2','own3','own4']
    ierr, idata = psspy.aflowint(sid, owner_brflow, ties_brflow, flag_brflow, istrings)
    if ierr:
        print('psspy.aflowint error = %d' % ierr)
        return
    iflow = array2dict(istrings, idata)
    # Branch Flow Data - Real
    rstrings = ['amps','pucur','pctrate','pctratea','pctrateb','pctratec','pctmvarate',
                'pctmvaratea','pctmvarateb',#'pctmvaratec','fract1','fract2','fract3',
                'fract4','rate','ratea','rateb','ratec',
                'p','q','mva','ploss','qloss',
                'o_p','o_q','o_mva','o_ploss','o_qloss'
                ]
    ierr, rdata = psspy.aflowreal(sid, owner_brflow, ties_brflow, flag_brflow, rstrings)
    if ierr:
        print('psspy.aflowreal error = %d' % ierr)
        return
    rflow = array2dict(rstrings, rdata)
    # Branch Flow Data - Complex
    xstrings = ['pq','pqloss','o_pq','o_pqloss']
    ierr, xdata = psspy.aflowcplx(sid, owner_brflow, ties_brflow, flag_brflow, xstrings)
    if ierr:
        print('psspy.aflowcplx error = %d' % ierr)
        return
    xflow = array2dict(xstrings, xdata)
    # Branch Flow Data - Character
    cstrings = ['id','fromname','fromexname','toname','toexname','nmetername','nmeterexname']
    ierr, cdata = psspy.aflowchar(sid, owner_brflow, ties_brflow, flag_brflow, cstrings)
    if ierr:
        print('psspy.aflowchar error = %d' % ierr)
        return
    cflow = array2dict(cstrings, cdata)
    return ibuses, rbuses, xbuses, cbuses, iplants, rplants, xplants, cplants, iloads, rloads, xloads, cloads, iswsh, rswsh, xswsh, cswsh,iflow, rflow, xflow, cflow
       

def GenInfo(iplants,rplants): 
    ngen = len(iplants['number'])
    
    ngen_AreaID = np.zeros(nsub)# generator in WOA
    IDGen_AreaID = [[] for _ in range(nsub)]
    
    for idx in range(0, ngen):
        id_Area = iplants['area'][idx]
        ngen_AreaID[id_Area-1]+=1
        IDGen_AreaID[id_Area-1].append(idx)
    
    pf = np.zeros((ngen,nsub))
    
    # for i in range(nsub):
    #     if i == 0 and ngen_AreaID[i]>1:
    #        pf[IDGen_AreaID, i] = 1 / (ngen_AreaID[i] - 1)
    #     elif i>0 and ngen_AreaID[i] > 0:
    #         pf[IDGen_AreaID, i] = 1 / (ngen_AreaID[i])
    #     pf[iplants['type'].index(3),i] = 0

    for i in range(nsub):
        for idx in IDGen_AreaID[i]:
            pf[idx, i] = 1 / (ngen_AreaID[i] - 1 if i == 0 else ngen_AreaID[i])
        pf[iplants['type'].index(3),i] = 0

    #pmax = np.zeros(ngen, dtype=float)
    #pmin = np.zeros(ngen, dtype=float)
    #for idx in range(0, ngen):
        #pmax[idx] = rplants['pmax'][idx]
        #pmin[idx] = rplants['pmin'][idx]   

    pmax = np.array(rplants.get('pmax', np.zeros(ngen)))
    pmin = np.array(rplants.get('pmin', np.zeros(ngen)))
	
    return ngen, ngen_AreaID, pf, pmax, pmin, IDGen_AreaID   
    
def BusinAreaID(ibuses):
    bus_AreaID=[]
    for i, bus in enumerate(ibuses['number']):
        if ibuses['area'][i]==AreaID:
           bus_AreaID.append(bus)  
    return bus_AreaID               

def BrchinAreaID(iflow,bus_AreaID):
    brch_list = [] 

    nbrh = len(iflow['fromnumber'])
    for idx in range(0, nbrh):
        brcht = [iflow['fromnumber'][idx],iflow['tonumber'][idx]]
        
        if iflow['tonumber'][idx] < 10000000 and iflow['fromnumber'][idx] < iflow['tonumber'][idx] and brcht not in brch_list:  # don't process 3-wdg xmer star-point buses
            if brcht[0] in bus_AreaID and brcht[1] in bus_AreaID:
                brch_list.append(brcht)  
    return brch_list        

# output frombus and tobus of branches            
def BusNo_brch():
    istrings = ['fromnumber','tonumber']
    ierr, idata = psspy.aflowint(sid, owner_brflow, ties_brflow, flag_brflow, istrings)
    if ierr:
        print('psspy.aflowint error = %d' % ierr)
        return
    FTBus_brch = array2dict(istrings, idata)
    return FTBus_brch
    
# output tie line index 
def FindTieline(FTBus_brch,from_bus_tline,to_bus_tline):
    # number of subsystems
    nbrh = len(FTBus_brch['fromnumber'])

    brch_list = []
    tie_line_indices = []

    # for idx in range(nbrh):
        # from_bus = FTBus_brch['fromnumber'][idx]
        # to_bus = FTBus_brch['tonumber'][idx]
        # brcht = [from_bus, to_bus]
        
        # if brcht not in brch_list:
            # for fbus, tbus in zip(from_bus_tline, to_bus_tline):
                # if from_bus == fbus and to_bus == tbus:
                    # tie_line_indices.append(idx)  # Store the index of the tie line
                    # brch_list.append(brcht)
                    # break   
                    
    nsub = len(from_bus_tline)
    
    for isub in range(nsub):
        brch_list_isub = []
        tie_line_indices_isub = []
        for fbus, tbus in zip(from_bus_tline[isub], to_bus_tline[isub]):  
             for idx in range(nbrh):
                 from_bus = FTBus_brch['fromnumber'][idx]
                 to_bus = FTBus_brch['tonumber'][idx]
                 brcht = [from_bus, to_bus]
                 
                 if brcht not in brch_list_isub and from_bus == fbus and to_bus == tbus:
                     #if from_bus == fbus and to_bus == tbus: 
                    tie_line_indices_isub.append(idx)
                    brch_list_isub.append(brcht)
                    break 
        tie_line_indices.append(tie_line_indices_isub)

    return tie_line_indices
 
def Update_plants(sid, flag_plant):
    istrings = ['number','type','area','zone','owner','dummy', 'status','ireg']
    ierr, idata = psspy.agenbusint(sid, flag_plant, istrings)
    if ierr:
        print('psspy.agenbusint error = %d' % ierr)
        return
    iplants = array2dict(istrings, idata)    
    
    rstrings = ['base','pu','kv','angle','angled','iregbase','iregpu','iregkv','vspu','vskv','rmpct',
            'pgen',  'qgen',  'mva', 'percent', 'pmax',  'pmin',  'qmax',  'qmin',  'mismatch',
            'o_pgen','o_qgen','o_mva','o_pmax','o_pmin','o_qmax','o_qmin','o_mismatch']
    ierr, rdata = psspy.agenbusreal(sid, flag_plant, rstrings)
    if ierr:
        print('psspy.agenbusreal error = %d' % ierr)
        return
    rplants = array2dict(rstrings, rdata)
    
    return iplants, rplants
    
def dispatch_unit(iplants,pgent):
    ngen = len(iplants['number'])
    for idx in range(0, ngen):
        # bus id
        busi = iplants['number'][idx]
        psspy.machine_chng_4(busi,r"1",realar1=pgent[idx])    
    

# collect real power output of generators, slack bus, and power flow of tie lines  
def RT_pgent(iplants,tie_line_indices):
    # import psspy

    # psspy.psseinit()

    psspy.fdns([0,0,0,1,1,1,99,0]) 
    
    sid = -1
    # generators
    ngen = len(iplants['number'])
    pgentt= psspy.amachreal(sid, 1,'PGEN')
    # # index of slack bus in generator array
    # IDSlack = iplants['type'].index(3)
    # #Pslackt= [pgentt[1][0][iplants['type'].index(3)],0,0]# 3 denotes slack bus; slack bus is in subsysem 0
    # IDAreaSlack = iplants['area'][IDSlack]
    # Pslackt = np.zeros(3)
    # Pslackt[IDAreaSlack-1] = pgentt[1][0][IDSlack]
    Pslackt= [0,0,pgentt[1][0][iplants['type'].index(3)]]# 3 denotes slack bus; slack bus is in subsysem 0
    pgent0= [pgentt[1][0][i] for i in range(0,ngen)] 
    
    # branch flow
    flag_brflow=1
    ierr, p_LF = psspy.aflowreal(sid, 1, 5, flag_brflow, 'p')
    p_TLF_temp = []
    for isub in range(nsub):
        p_TLF_isub = [p_LF[0][tie_line_indices[isub][i]] for i in range(len(tie_line_indices[isub]))]
        p_TLF_temp.append(p_TLF_isub)
        
    # Find the maximum length of any sublist
    max_length = max(len(sublist) for sublist in p_TLF_temp)    
    padded_p_TLF = [sublist + [0] * (max_length - len(sublist)) for sublist in p_TLF_temp]

    # Convert to a NumPy array
    p_TLF0 = np.array(padded_p_TLF)
    
    return np.array(pgent0),np.array(Pslackt), np.array(p_TLF0)     
    #return np.array(pgent0),Pslackt, np.array(p_TLF0)

# def AGC_WOA(pgent0, pgent, ngen_AreaID, pmin, pmax, pf, dPslack, iplants,Pslack0,p_TLF0,IDGen_AreaID):
    # ############ AGC logic 
    # thr_converge, thr_noloss, thr_pf = 0.2, 0.2, 0.01 
    # while abs(dPslack)>thr_converge and sum(pf)>thr_pf: 
        # err_noloss = dPslack 
        # while abs(err_noloss)>thr_noloss and sum(pf)>thr_pf:
            # # update participation factor
            # pf = pf/sum(pf)  
            # print('Number of dispatchable units:',np.count_nonzero(pf))
            # #for idx in range(0, ngen_AreaID):
            # for idx in IDGen_AreaID:
                # if pf[idx]!=0:
                    # # estimated real power output
                    # pgeni=pgent[idx]+pf[idx]*err_noloss
                    
                    # # project output within bounds              
                    # pgent[idx] = min(pmax[idx],max(0,max(pmin[idx], pgeni)))              
                    # #residuali = (pgeni-pgent[idx]) 
                    
                    # # if idx gen hit the bounds
                    # if abs(pgent[idx]-pmin[idx])<0.001 or abs(pgent[idx]-pmax[idx])<0.001 or abs(pgent[idx])<0.001:
                       # pf[idx]=0# gen idx will not be dispatched any more
          
            # ## dispatch enough power?
            # dif_pgen = pgent-pgent0
            # dif_pgen[iplants['type'].index(3)] = 0# slack bus cannot be dispatched
            # err_noloss = dPslack-sum(dif_pgen)# mismatch between target (dPslack) and current output                  
            
        # # mismatch of slack bus on AC power flow   
        # dispatch_unit(iplants,pgent)     
        # pgent0,Pslackt,p_TLF=RT_pgent(iplants,tie_line_indices)
        # dPslack = Pslackt-Pslack0-(sum(p_TLF-p_TLF0))
     
         
    # if abs(dPslack)>thr_converge:
       # success = 0  
       # print('Not enough power generation from the control system!')           
    # else:
        # # dispatch real power output of unit
        # dispatch_unit(iplants,pgent)
        # # run power flow        
        # psspy.fdns([0,0,0,1,1,1,99,0])  
        # success = 1        
        # print('Success!')

 
    # return pgent, success
# calculate generator setpoints dispatched by AGC 
def AGC_logic(err_noloss,thr_noloss,pf_i,thr_pf,IDGen_AreaID_i,pgent,pmax,pmin,iplants,dPslack_i):     
    pgent0 = pgent.copy()    
    while abs(err_noloss)>thr_noloss and sum(pf_i)>thr_pf:
        # update participation factor
        pf_i = pf_i/sum(pf_i)  
        print('Number of dispatchable units:',np.count_nonzero(pf_i))
        
        
        #for idx in range(0, ngen_AreaID):
        for idx in IDGen_AreaID_i:
            if pf_i[idx]!=0:
                # estimated real power output
                pgeni=pgent[idx]+pf_i[idx]*err_noloss
                
                # project output within bounds              
                pgent[idx] = min(pmax[idx],max(0,max(pmin[idx], pgeni)))              
                #residuali = (pgeni-pgent[idx]) 
                
                # if idx gen hit the bounds
                if abs(pgent[idx]-pmin[idx])<0.001 or abs(pgent[idx]-pmax[idx])<0.001 or abs(pgent[idx])<0.001:
                   pf_i[idx]=0# gen idx will not be dispatched any more
      
        ## dispatch enough power?
        dif_pgen = pgent-pgent0
        dif_pgen[iplants['type'].index(3)] = 0# slack bus cannot be dispatched
        err_noloss = dPslack_i-sum(dif_pgen)# mismatch between target (dPslack) and current output  
    return pgent[IDGen_AreaID_i],pf_i
 
# output new setpoints of generators, convergence status of AGC dispatch, mismatch between contracted and actual setpoints 
def AGC(ngen,pgent, ngen_AreaID, pmin, pmax, pf, dPslackt, iplants,Pslack0,p_TLF0,IDGen_AreaID):
    ############ AGC logic 
    thr_converge, thr_noloss, thr_pf, iter_max, iteration = 0.2, 0.2, 0.01, 100, 0 
    pgent_i = np.zeros(ngen)
    proceed = 1
    while (proceed>=1):
        print('Iteration:',iteration)
        for i in range(nsub):
            pf_i = pf[:,i]
            err_noloss = dPslackt[i]
            IDGen_AreaID_i = IDGen_AreaID[i]
            dPslack_i = dPslackt[i]
            if i==2:
               print('i')
            pgent_i[IDGen_AreaID_i],pf[:,i]=AGC_logic(err_noloss,thr_noloss,pf_i,thr_pf,IDGen_AreaID_i,pgent,pmax,pmin,iplants,dPslack_i)  
          
        pgent = pgent_i
        
        # dispatch  
        # dispatchable generators
        dispatch_unit(iplants,pgent)     
        # update power flow
        pgent,Pslackt,p_TLFt=RT_pgent(iplants,tie_line_indices)
        # update ACE
        dPslackt= ACE(Pslack0,p_TLF0,Pslackt,p_TLFt) 
        
        proceed = 0
        for i in range(nsub):
            if abs(dPslackt[i])>thr_converge and sum(pf[:,i])>thr_pf:
               proceed+=1
         
        iteration+=1         
         
    if max(abs(dPslackt))>thr_converge:
       success = 0  
       print('ACE cannot be 0!')           
    else:
        # dispatch real power output of unit
        dispatch_unit(iplants,pgent)
        # run power flow        
        psspy.fdns([0,0,0,1,1,1,99,0])  
        success = 1        
        print('Success!')

    return pgent, success, dPslackt
    
# calculate Area control error  
def ACE(Pslack0,p_TLF0,Pslackt,p_TLFt):
    dPslackt=np.zeros(nsub)
    for i in range(nsub):
        dPslackt[i] = (Pslackt[i]-sum(p_TLFt[i]))-(Pslack0[i]-sum(p_TLF0[i]))
    return dPslackt

# collect new setpoints of generators, convergence status of AGC dispatch, mismatch between contracted and actual setpoints
def update_AGC(Pslack0,p_TLF0):
    #psspy.fdns([0,0,0,1,1,1,0,0])
    # run power flow
    # get tie_line index after trip a line
    FTBus_brch = BusNo_brch()
    tie_line_indices = FindTieline(FTBus_brch,from_bus_tline,to_bus_tline)
    
    # update iplants
    iplants, rplants = Update_plants(sid, flag_plant)  

    ngen, ngen_AreaID, pf, pmax, pmin, IDGen_AreaID = GenInfo(iplants,rplants)      
    
    # update power flow 
    pgent,Pslackt,p_TLFt=RT_pgent(iplants,tie_line_indices)
    # calculate Area control error
    dPslackt= ACE(Pslack0,p_TLF0,Pslackt,p_TLFt)

    
    # collect new setpoints of generators, convergence status of AGC dispatch, mismatch between contracted and actual setpoints
    pgent_t,  AGC_success_t,dPslackt_end_t = AGC(ngen,pgent, ngen_AreaID, pmin, pmax, pf, dPslackt, iplants,Pslack0,p_TLF0,IDGen_AreaID)
    return pgent_t, AGC_success_t, dPslackt_end_t
        
def ACC_report(i):
    # run power flow
    psspy.fnsl([1,1,1,1,1,0,0,0])
    
    psspy.dfax_2([0,1,0],acc_sub,acc_mon, acc_con,r"""C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code\WOA""")
    psspy.accc_with_dsp_3(0.5,[0,0,0,1,1,2,0,0,0,0,0],r"""WOA""",acc_dfx,acc_acc,
    acc_thr,"","")
    psspy.report_output(2,acc_output+str(i)+".txt",[0,0])
    psspy.accc_single_run_report_6([0,1,1,1,1,1,1,0,1,1,1,0,0,0,0],[0,0,0,0,6000],[0.5,5.0,100.0,0.0,0.0,0.0,99999.,0.0],"",acc_acc)
    
# ====================================================================================================
# ====================================================================================================
if __name__ == '__main__':

    import psse35
    AreaID = 1
    # Power flow executes every 4 seconds, AGC every minute, and ACC every 5 minutes
    dt_AGC = 15# AGC update frequency is 1/dt_AGC of load update frequency
    dt_ACC = 15*5# ACC update frequency is 1/dt_ACC of load update frequency 


    import psspy

    psspy.psseinit()
    
    # simulation parameter
    if psspy.bsysisdef(0):
        sid = 0
    else:   # Select subsytem with all buses
        sid = -1

    flag_bus     = 1    # in-service
    flag_plant   = 1    # in-service
    flag_load    = 1    # in-service
    flag_swsh    = 1    # in-service
    flag_brflow  = 1    # in-service
    owner_brflow = 1    # use bus ownership, ignored if sid is -ve
    ties_brflow  = 5    # ignored if sid is -ve    
    
    # path of input and output files for ACC report (n-1 contingency analysis)
    acc_mon = r"""C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code\WOA.mon"""
    acc_sub = r"""C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code\WOA.sub"""
    acc_con = r"""C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code\WOA.con"""
    acc_dfx = r"""C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code\WOA.dfx"""
    acc_acc = r"""C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code\WOA.acc"""
    acc_thr = r"""C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code\WOA.thr"""
    acc_output = r"C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code\acc_report\acc_report(118)_minute"
    
    
    
    ############# system parameters
    ## pout_report(path for IEEE 118-bus sav file, path for output file,subsystem index), subsystem index is not used in the function
    ibuses, rbuses, xbuses, cbuses, iplants, rplants, xplants, cplants, iloads, rloads, xloads, cloads, iswsh, rswsh, xswsh, cswsh,iflow, rflow, xflow, cflow=pout_report(r'C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code\IEEE118(3A).sav',r'C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code',AreaID)# report the results of xxx.sav; 
       
    # Tie line index 
    from_bus_tline = [[15,19,23,30],[33,34,38,47,49,65],[24,69,69,68]]# from-bus index
    to_bus_tline = [[33,34,24,38],[15,19,30,69,69,68],[23,47,49,65]]# to-bus index
    
    tie_line_indices = FindTieline(iflow,from_bus_tline,to_bus_tline)# iflow['fromnumber']/iflow['tonumber'] collects the from_bus/to_bus across all the branches; 
    nsub = len(from_bus_tline)
    # ########## Initial state   
    ## collect real power output of generators, slack bus, and power flow of tie lines    
    pgent0,Pslack0,p_TLF0=RT_pgent(iplants,tie_line_indices)
    pgent = pgent0.copy()
    
    # Get area id (0,1, or 2) across all the buses
    bus_AreaID = BusinAreaID(ibuses)
    # Get area id across all the branches    
    brch_AreaID0 = BrchinAreaID(iflow,bus_AreaID)

    ################ update demand in WOA
    # pload0 = xloads['mvaact'][0].real
    # ploadt = pload0+70
    # Loadbus = iloads['number'][0]# load bus
    # psspy.load_chng_6(Loadbus,r'1',realar1=ploadt)
   
    path_input = r'C:\Users\HUANJ\Desktop\work\Projects\HumanFactor\Code\Input\TimeSeriesLoad(SAV)'
    
    # Get all .sav files from the directory
    sav_files = [file for file in os.listdir(path_input) if file.endswith('.sav')]
    sav_files.sort()  # Sort files to ensure correct order if needed
    num_files = len(sav_files)  # Number of .sav files
    
    
    converge=[]
    #AGC_converge = np.zeros(num_files)
    AGC_success= np.zeros(int(num_files/dt_AGC))
    dPslackt_end= np.zeros((num_files,3))
    pgent_end = np.zeros((num_files,len(iplants['number'])))
    
    # place holder for tripping a line
    pgent_sw  = pgent0.copy()
    AGC_success_sw= np.zeros(num_files)
    dPslackt_end_sw= np.zeros((num_files,3))
    pgent_end_sw = np.zeros((num_files,len(iplants['number'])))
    
    start_time = time.time()
    #for i in range(1,num_files):
    for i, filename in enumerate(sav_files):
        # update load profile
        psspy.case(os.path.join(path_input, filename))
        # run power flow
        psspy.fnsl([0,0,0,1,1,1,0,0])
        # collect convergence status of power flow
        converge.append(psspy.solved())
        print('Power flow converge (0) or not (non-zero):',psspy.solved())
        
        # update AGC every minute
        if i % dt_AGC == 0:
           # collect new setpoints of generators, convergence status of AGC dispatch, mismatch between contracted and actual setpoints
           pgent,  AGC_success[i],dPslackt_end[i,:] = update_AGC(Pslack0,p_TLF0)   
  
        pgent_end[i,:] = pgent
        
        # # # ACC report
        if i % dt_ACC == 0:
           # execute N-1 contingency analysis
           ACC_report(i)
        
        # # control action, like tripping a line
        # ################ trip a line in WOA
        # id_tripline = 2
        # psspy.branch_chng_3(brch_AreaID0[id_tripline][0],brch_AreaID0[id_tripline][1],r"1",intgar1=0)
        
        # psspy.fdns([0,0,0,1,1,1,0,0])
        # # # run power flow        
        # pgent_sw,  AGC_success_sw[i],dPslackt_end_sw[i,:] = update_AGC(Pslack0,p_TLF0)
        # pgent_end_sw[i,:] = pgent_sw
        
        # # switch back the line
        # psspy.branch_chng_3(brch_AreaID0[id_tripline][0],brch_AreaID0[id_tripline][1],r"1",intgar1=1)

    end_time = time.time()
    # Calculate the total computation time
    computation_time = end_time - start_time 

    print('Computational time:{:.2f} seconds'.format(computation_time))    
  
    print("The number of diverged cases:", sum(converge))
    
    print("The number of cases when AGC fails:",num_files-sum(AGC_success))

    #print("The number of cases when AGC fails after tripping a line:",num_files-sum(AGC_success_sw))
    # print(dPslackt_end)  
    # print('AGC_success')
    # print(AGC_success)    
    # # print('AGC_success_sw')
    # # print(AGC_success_sw)

# ====================================================================================================
