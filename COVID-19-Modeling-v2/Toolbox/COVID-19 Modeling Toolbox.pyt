# -*- coding: utf-8 -*-

import arcpy
import arcpy as ARCPY
import arcgisscripting as arc
import SSUtilities as utils
import SSDataObject as SSDO

from datetime import datetime as dt
from datetime import date
import pandas as pd
from pandas import DataFrame
import numpy as np
import numpy as NUM
import os
import os as OS
import datetime
import arcpy.management as DM
import arcpy.da as DA
import ErrorUtils as ERROR

from chime_develop_112.src.penn_chime.models import  SimSirModel as Model
from chime_develop_112.src.penn_chime.parameters import Parameters
from chime_develop_112.src.penn_chime.parameters import Disposition as RateDays
import locale as LOCALE
LOCALE.setlocale(LOCALE.LC_ALL, '')

import pdb

numpyDtypeConvert = {NUM.dtype('float64'):"DOUBLE",
                     NUM.dtype('int32'): "LONG",
                     NUM.dtype('int64'): "LONG",
                     NUM.float64 : "DOUBLE",
                     NUM.int32: "LONG",
                     NUM.int64: "LONG",
                     NUM.dtype('O'): "DATE",
                     int: "LONG",
                     float: 'DOUBLE',
                     '<U': "TEXT"}

def numpyToFieldType(typeNumpy):
    """
    Convert Numpy dtype to Candidate Type Supported
    INPUT:
        typeNumpy (dtype): type array
    OUTPUT:
        return (str) : Supported (LONG, TEXT..)
    """
    if '<U' in str(typeNumpy):
        typeNumpy = '<U'
    if '<M' in str(typeNumpy):
        typeNumpy = NUM.dtype('O')
    if 'datetime' in str(typeNumpy):
        typeNumpy = NUM.dtype('O')

    return numpyDtypeConvert[typeNumpy]

class DataContainer(object):
    """
    This Data Container allows to write point/polygon - table features class using 
    the ARC method
    INPUT:
        spatialRef {spatialReference: None}: Feature class, None-> table
        xy {2d Array}: Coordinates
        shapes {list shapes}: List of shapes
        z {1d Array}: Z Coordinate

    Note:
        TODO: Support Polyline
    """
    def __init__(self, spatialRef = None, xy = None, shapes = None, z= None):
        if spatialRef is not None:
            if xy is not None:
                self.xyCoords = xy
                self.numObs = len(xy)
                if z is not None:
                    self.zCoords = z
            else:
                self.xyCoords = None
                if shapes is not None:
                    self.numObs = len(shapes)

        self.spatialRef = spatialRef
        self.requireGeometry = False

        #### Point 0 / Polyline 1/ Polygon 2 ####
        self.renderType = 0 

        #### Eval Z elevation ####
        if z is not None:
            self.hasZ = True
        else:
            self.hasZ = False

        self.shapes = None

        if spatialRef is not None:
            if shapes is not None and type(shapes) == list:
                self.requireGeometry = True
                if "Polygon" in str(type(shapes[0])):
                    self.renderType = 2
                else:
                    self.renderType = 1
                self.shapes = shapes

    def createFieldsFromArrays(self, listArrays, outputFC, names = None, aliasNames = None):
        """ Create a list of candidate fields from a list of Numpy Arrays or
            list of Candidate Fields (Check field Name)
        INPUT:
            listArray (List Arrays/CandidateFields): Each element in the list 
                                                     will be a field in outputFC
            outputFC (str): Output path
            names {list str}: Name Fields (used when listArray just contains arrays)
            aliasNames {list str}: Field Alias 
        OUTPUT:
            output: List of CandidateFields
        """

        if len(listArrays) > 0:
            if "CandidateField" in str(type(listArrays[0])):
                return listArrays
            #### Get Names ####
            nameList = []
            nameAlias = []

            if aliasNames is not None:
                nameAlias = aliasNames

            if names is not None:
                nameList = names
            else:
                #### If Names is Provided ####
                nameList = ["Field" + str(i+1) for i in range(len(listArrays))]

            typeList = [type(arr) for arr in listArrays]

            pDtype = [str(type(arr)) for arr in listArrays if 'ndarray' in str(type(arr))]

            if len(pDtype):
                typeList = [arr.dtype for arr in listArrays]

                fields = None
                if len(nameAlias) == 0:
                    fields = [SSDO.CandidateField(name = nameList[id], type = numpyToFieldType(arr.dtype), data = arr)
                         for id, arr in enumerate(listArrays)]
                else:
                    fields = [SSDO.CandidateField(name = nameList[id], alias = nameAlias[id],
                                                 type = numpyToFieldType(arr.dtype), data = arr)
                         for id, arr in enumerate(listArrays)]
                #### Check Output Field Names ####
                fields = checkCandidateFieldName(fields, outputFC)
                return fields

            #### Check If The List Contains Arrays ####
            if list in typeList:
                arrN = []
                for i in listArrays:
                    if type(i) == list:
                        arr = NUM.array(i)
                        arrN.append(arr)
                    elif "ndarray" in str(type(i)):
                        arrN.append(i)
                    else:
                        ARCPY.AddError("Should be a list/ ndarray")
                        raise SystemExit
                listArrays = arrN

                ### Get Type of Each Array ####
                typeList = [numpyToFieldType(arr.dtype) for arr in listArrays]

            fields = None
            if len(nameAlias) == 0:
                fields = [SSDO.CandidateField(name = nameList[id], type = typeList[id], data = arr)
                         for id, arr in  enumerate(listArrays)]
            else:
                fields = [SSDO.CandidateField(name = nameList[id], alias = nameAlias[id], 
                                              type = typeList[id], data = arr)
                         for id, arr in  enumerate(listArrays)]
            #### Check Output Field Names ####
            fields = checkCandidateFieldName(fields, outputFC)
            return fields
        else:
            return []


    def generateOutput(self, outputFC, listFields1, names = None, alias = None):
        """ Create Output Feature Class / table, Depend type of container

        outputFC {str} : output path
        listFields {list CandidateFields/arrays}: Candidate Fields List or []
        names {list str}: List Names
        """
        listFields= self.createFieldsFromArrays(listFields1, outputFC, names, alias)
        ERROR.checkOutputPath(outputFC)
        outPath, outName = OS.path.split(outputFC)
        mFlag = "DISABLED"
        zFlag = "DISABLED"

        if self.hasZ:
            zFlag = "ENABLED"

        if self.shapes is not None:

            try:
                DM.CreateFeatureclass(outPath, outName, "POLYGON", 
                                      "", mFlag, zFlag, 
                                      self.spatialRef)
            except:
                ARCPY.AddIDMessage("ERROR", 210, outputFC)
                raise SystemExit()
        if self.shapes is None:
            try:
                DM.CreateFeatureclass(outPath, outName, "POINT", 
                                      "", mFlag, zFlag, 
                                      sself.spatialRef)
            except:
                ARCPY.AddIDMessage("ERROR", 210, outputFC)
                raise SystemExit()

        #### Add Fields to Output FC ####
        shapeFieldNames = ["SHAPE@"]
        for field in listFields:
            utils.addEmptyField(outputFC, field.name , field.type, field.alias)

        rows = DA.InsertCursor(outputFC, shapeFieldNames + [field.name for field in listFields])
        ARCPY.SetProgressor("step", "Writting features..", 0, self.numObs, 1)

        for id in NUM.arange(self.numObs):
            ARCPY.SetProgressorLabel("Writting feature {0}...".format(id+1))
            rowResult = []

            if self.shapes is not None:
                rowResult.append(self.shapes[id])
            else:
                values = self.xyCoords[id]
                if not self.hasZ:
                    pnt = (values[0], values[1], 0)
                else:
                    valueZ= self.zCoords[id]
                    pnt = (values[0], values[1], valueZ)
                rowResult.append(pnt)

            for i, field in enumerate(listFields):
                if field.type.upper() == "DATE":
                    rowResult.append(np.datetime_as_string(field.data[id], unit='D'))
                else:
                    rowResult.append(field.data[id])

            rows.insertRow(rowResult)
            ARCPY.SetProgressorPosition()
        del rows

def isGDB(input):
    """Returns whether the input feature class is contained in 
    a gdb, robust for feature layer, input and output fc

    INPUTS:
    input (str): feature layer (string), fc input, fc output

    OUTPUT:
    return (bool): is the input in a gdb?
    """

    isContained = False
    path = input
    try:
        d = ARCPY.Describe(input)
        path = str(d.CatalogPath)
    except:
        pass

    try:
        path = path.upper()
        if ".GDB" in path:
            isContained = True
        else:
            if input[-3:].upper() == "SHP" :
                isContained = False
            else:
                if ".GDB" in input:
                    isContained = True

    except:
        pass
    return isContained

def checkCandidateFieldName(candidateFields, output):
    """ This function adjusts the field names for shp and dbf file outputs
        from candidate fields instances
    INPUT:
        candidateFields (list candidateFields): List Candidate Fields
        output (str): Output Path
    OUTPUT:
        cadidateFields: List Candidate Fields 
    """
    
    qualifyNames = ARCPY.env.qualifiedFieldNames
    isGdb = isGDB(output)

    invalidChar = "`~@#$%^&*()-+=|\,<>?/{}!'[]:;\n\r"

    for i in NUM.arange(len(candidateFields)):
        if '$.' in candidateFields[i].name:
            candidateFields[i].name = candidateFields[i].name.replace("$.","__")
        for e in invalidChar:
            candidateFields[i].name = candidateFields[i].name.replace(e,"_")

        if "." in candidateFields[i].name:
            if qualifyNames and isGdb:
                candidateFields[i].name = candidateFields[i].name.replace(".","_")
            elif qualifyNames and not isGdb:
                name, field = candidateFields[i].name.split('.')
                candidateFields[i].name = name
            elif not qualifyNames:
                name, field = candidateFields[i].name.split('.')
                candidateFields[i].name = field

    if isGdb:
        for i in NUM.arange(len(candidateFields)):
            if ".SHP" in candidateFields[i].name.upper():
                candidateFields[i].name = candidateFields[i].name.replace(".SHP","")
            if " " in candidateFields[i].name:
                candidateFields[i].name = candidateFields[i].name.replace(" ","_")

        return candidateFields
    else:

        for i in NUM.arange(len(candidateFields)):
            if ".SHP" in candidateFields[i].name.upper():
                candidateFields[i].name = candidateFields[i].name.replace(".SHP","")
            if " " in candidateFields[i].name:
                candidateFields[i].name = candidateFields[i].name.replace(" ","_")


        maxNumPos = 10
        reduceSizeTo = 7
        fields = [ i.name for i in candidateFields]
        maxLen = max(map(len, fields))

        #### Reduce Field Name Size ####
        nFields = NUM.array([(i[0:maxNumPos], i, False) if len(i) > maxNumPos else (i, i, False) 
                                for i in fields], 
                                dtype =[("newField", "U10"), 
                                        ('field', "U" +str(maxLen)),
                                        ('changed', bool)])
            
        #### Verify Uniqueness ####
        unique, count = NUM.unique(nFields['newField'], return_counts= True)

        #### Fix Uniqueness ####
        if len(nFields) != len(count):
            for indu, uniq in enumerate(unique):
                numRep = count[indu]
                cnt = 1
                if numRep > 1:
                    for index, field in enumerate(nFields):
                        if field[0] == uniq:
                            if nFields[index][2]:
                                fieldValue = field[0][0:(reduceSizeTo-1)] + '{:02}'.format(cnt) + field[0][-3:]
                                nFields[index][0] = fieldValue
                            else:
                                fieldValue = field[0][0:(reduceSizeTo+1)] + '{:02}'.format(cnt)
                                nFields[index][0] = fieldValue
                            cnt += 1

        #### Replace Candidate Field Names By Unique Names ####
        for idField in NUM.arange(len(candidateFields)):
            if candidateFields[idField].name == nFields[idField][1]:
                candidateFields[idField].name = nFields[idField][0]
        return candidateFields

def ContainerVersion():
    if hasattr(utils, 'DataContainer'):
        temp = utils.DataContainer()
        if 'alias' in temp.generateOutput.__code__.co_varnames:
            return utils.DataContainer
        else:
            return DataContainer
    else:
        return DataContainer

def setEnvSpatialReference(inputSpatialRef):
    """Returns a spatial reference object of Env Setting if exists.

    INPUTS:
    inputSpatialRef (obj): input spatial reference object

    OUTPUT:
    spatialRef (class): spatial reference object
    """

    envSetting = arcpy.env.outputCoordinateSystem
    if envSetting != None:
        #### Set to Environment Setting ####
        spatialRef = envSetting
    else:
        spatialRef = inputSpatialRef

    return spatialRef

def setEnvExtent(inputExtent):
    """Returns a spatial extent object of Env Setting if exists.

    INPUTS:
    inputExtent (obj): input spatial reference object

    OUTPUT:
    spatialExtent (class): spatial reference object
    """

    envSetting = arcpy.env.extent
    if envSetting != None:
        #### Set to Environment Setting ####
        spatialExtent = envSetting
    else:
        spatialExtent = inputExtent

    return spatialExtent

def getInputFields(*argv):
    """
    Returns a list of all user defined input fields 
    Inputs: Field names from the GP tool
    """
    fieldList = [name.upper() for name in argv[0] if name]

    return fieldList

def setInputMatrix(ssdo, fieldArray, constantArray, colNames):
    """
    Utility function to merge contant and field inputs
    into a one data matrix
    Inputs: 
    - SSDataObject instance
    - Array containing field-based parameter inputs
    - Array containing constant parameter inputs
    Output:
    - ND-Array containing input data
    """
    out_df = pd.DataFrame(columns = colNames)
    n = ssdo.numObs
    for varName, field, constVal in zip(colNames, fieldArray, constantArray):
        if field:
            out_df[varName] = ssdo.fields[field.upper()].data
        elif constVal:
            if constVal is None:
                arcpy.AddMessage("Please provide a valid value for variable {0}".format(varName));
                raise SystemExit()
            out_df[varName] = np.tile(utils.strToFloat(constVal), n)
        else:
            arcpy.AddMessage("A model parameter has no field or constant input")
            raise SystemExit()
    
    return out_df

def updateArray(arrays, index, slc, df, outputFields, typeOutputFields, arrSum = None):
    for name, type in zip(outputFields,typeOutputFields) :
        if type == "DATE":
            arrays[index][slc] = [e for e in df[name].copy()]
            if arrSum is not None:
                arrSum[index] = arrays[index][slc]
        if type == "LONG":
            arrays[index][slc] = np.asarray(df[name].copy(), dtype = np.float32)
            if arrSum is not None:
                arrSum[index] += arrays[index][slc]
        if type == "DOUBLE":
            arrays[index][slc] =np.asarray(df[name].copy(), dtype = np.float32)
            if arrSum is not None:
                arrSum[index] += arrays[index][slc]
        index +=1
    return index

def typeNewFieldsBasicStats():
    fieldNames = ["pk_hsp", 
                  "pk_day_hsp", 
                  "pk_dte_hsp",
                  "pk_icu", 
                  "pk_day_icu", 
                  "pk_dte_icu" , 
                  "pk_vnt",
                  "pk_day_vnt",
                  "pk_dte_vnt"]

    fieldAlias =["Peak Hospitalized Census",
                 "Peak Day for Hospitalized Census",
                 "Peak Date for Hospitalized Census",
                 "Peak ICU Census",
                 "Peak Day for ICU Census",
                 "Peak Date for ICU Census",
                 "Peak Ventilated Census",
                 "Peak Day for Ventilated Census",
                 "Peak Date for Ventilated Census"
                ]
    fieldTypes = ["DOUBLE",
                  "LONG",
                  "DATE", 
                  "DOUBLE", 
                  "LONG", 
                  "DATE", 
                  "DOUBLE",
                  "LONG", 
                  "DATE"]
    return fieldNames, fieldTypes, fieldAlias

def basicStats(df , nameH="hospitalized", nameI = "icu", nameV = "ventilated" ):

    maxH = df[nameH].max()
    maxI = df[nameI].max()
    maxV = df[nameV].max()

    idH = NUM.where(df[nameH]==maxH)[0]
    if type(idH) == np.ndarray:
        if len(idH) > 0:
            idH = idH[0]
        else:
            return [None]*9
    dateH = df["date"][idH]

    idI= NUM.where(df[nameI]==maxI)[0][0]
    if type(idI) == np.ndarray:
        idI= idI[0]
    dateI = df["date"][idI]

    idV= NUM.where(df[nameV]==maxV)[0][0]
    if type(idV) == np.ndarray:
        idV= idV[0]
    dateV = df["date"][idV]

    return maxH, idH, dateH, maxI, idI, dateI, maxV, idV, dateV 

def typeNewFieldsBasicStatsOver(nAbr =[]):

    fieldNames =[]
    fieldTypes = []
    fieldAlias = []
    for abrv in nAbr:
        fn,ft, fa = namesCap(abrv)
        fieldNames.extend(fn)
        fieldTypes.extend(ft)
        fieldAlias.extend(fa)

    return fieldNames, fieldTypes, fieldAlias

completeName = {"hos":"Hospitalized",
                 "icu":"ICU",
                 "vnt":"Ventilated"}

def namesCap(abrv):
    name = completeName[abrv]
    fieldNames = ["oc_{0}_max",
                  "oc_{0}_day", 
                  "oc_{0}_dte",
                  "oc_{0}_pct",
                  "oc_{0}_num" ]
    fieldAlias = ["Over Capacity {0} Maximum Number",
              "Over Capacity {0} Day", 
              "Over Capacity {0} Date",
              "Over Capacity {0} Maximum Percent",
              "Over Capacity {0} Number of Days" ]

    fieldNames = [ i.format(abrv) for i in fieldNames]
    fiedlAlias = [ i.format(name) for i in fieldAlias]
    fieldTypes = ["LONG",
                  "LONG", 
                  "DATE", 
                  "DOUBLE", 
                  "LONG"]
    
    return fieldNames, fieldTypes, fieldAlias


def overCapStats(df, index, ssdo, fieldName,refNameDF, duration ):
    maxH = df[refNameDF].max()
    value = ssdo.fields[fieldName.upper()].data[index]
    prc = np.nan
    if value > 0:
        prc = (maxH - value)*100 / value


    maskOver = df[refNameDF] > value
    reachMax = -1
    vMax = NUM.where(maskOver)[0]

    if len(vMax)==0:
        return -1, reachMax, dt.strptime(dummy_date, "%m/%d/%Y"), -1, -1

    numberDaysMax = 0
    dMax = df['date'][0]
    reachMax = vMax[0]

    if reachMax > -1:
        dMax = df['date'][reachMax]

    if type(dMax) == np.ndarray:
        dMax = dMax[0]

    if len(vMax):
        reachMax = vMax[0]
        numberDaysMax = len(vMax)

    return maxH-value, reachMax, dMax, prc, numberDaysMax

def addArrays(arrays, size, fieldTypeOutput, origFieldsDtype, fieldsNameOutput):
    for id, type in enumerate(fieldTypeOutput):
        if type in ["SINGLE", "FLOAT", "DOUBLE"]:
            arrays.append(NUM.zeros(size, float))
        if type == "DATE":
            arrays.append(NUM.empty(size, 'datetime64[s]'))
        if type in ["SHORT","LONG"]:
            arrays.append(NUM.empty(size, NUM.int32))
        if type == "TEXT":
            arrays.append(NUM.empty(size,origFieldsDtype[fieldsNameOutput[id]]))


def checkValue(parameter,  minValue = 0 , maxValue = 100, onlyCheckLowerBound = False, defaultVal = None, fieldParameter = None, minExclude = False, maxExclude = False):
    if parameter.value:
        if parameter.value == field_str:
            if fieldParameter:
                if fieldParameter.value is None:
                    if defaultVal is not None:
                        parameter.value = str(defaultVal)
                    else:
                        return
                else:
                    return
            else:
                return
        value = None

        try:
            value = utils.strToFloat(parameter.value)
        except:
            if defaultVal is not None:
                parameter.value = str(defaultVal)
            else:
                parameter.setErrorMessage(fr"Value should be numeric")
            return

        if onlyCheckLowerBound:
            if minExclude:
                if value is not None and value <= minValue:
                    parameter.setErrorMessage(fr"{parameter.displayName} should be greater than {minValue}")
            else:
                if value is not None and value < minValue:
                    parameter.setErrorMessage(fr"{parameter.displayName} should be greater than or equal to {minValue}")
        else:
            if value is not None and value < minValue or value > maxValue:
                parameter.setErrorMessage(fr"{parameter.displayName} should be greater than or equal to {minValue} and less than or equal to {maxValue}")

def getAliasFields(fieldNames):
    alias = {'DblTime' : 'Doubling Time',
     'SocDistPer' : 'Social Distance %',
     'InfHospPer' : 'Infected Hospitalized %',
     'InfICUPer' : 'Infected ICU %',
     'InfVentPer' : 'Infected Ventilator %',
     'HospStay' : 'Hospital Stay',
     'ICUStay' : 'ICU Stay',
     'VentStay' : 'Ventilator Stay',
     'HospMrShrP' : 'Hospital Market Share %',
     'InfDays' : 'Infectious Days',
     'NumHosp' : 'Number of Hospitalized',
     'NumInf' : 'Number of Infections',
     'pk_hsp' : 'Peak Hospitalized Census',
     'pk_day_hsp' : 'Peak Day for Hospitalized Census',
     'pk_dte_hsp' : 'Peak Date for Hospitalized Census',
     'pk_icu' : 'Peak ICU Census',
     'pk_day_icu' : 'Peak Day for ICU Census',
     'pk_dte_icu' : 'Peak Date for ICU Census',
     'pk_vnt' : 'Peak Ventilated Census ',
     'pk_day_vnt' : 'Peak Day for Ventilated Census',
     'pk_dte_vnt' : 'Peak Date for Ventilated Census',
     'oc_hos_max' : 'Over Capacity Hospitalized Maximum Number',
     'oc_hos_day' : 'Over Capacity Hospitalized Day',
     'oc_hos_dte' : 'Over Capacity Hospitalized Date',
     'oc_hos_pct' : 'Over Capacity Hospitalized Maximum Percent ',
     'oc_hos_num' : 'Over Capacity Hospitalized Number of Days ',
     'oc_icu_max' : 'Over Capacity ICU Maximum Number',
     'oc_icu_day' : 'Over Capacity ICU Day',
     'oc_icu_dte' : 'Over Capacity ICU Date',
     'oc_icu_pct' : 'Over Capacity ICU Maximum Percent',
     'oc_icu_num' : 'Over Capacity ICU Number of Days',
     'oc_vnt_max' : 'Over Capacity Ventilated Maximum Number',
     'oc_vnt_day' : 'Over Capacity Ventilated Day',
     'oc_vnt_dte' : 'Over Capacity Ventilated Date',
     'oc_vnt_pct' : 'Over Capacity Ventilated Maximum Percent',
     'oc_vnt_num' : 'Over Capacity Ventilated Number of Days',
     'new_hosp' : 'New Hospitalizations ',
     'new_icu' : 'New ICU Admissions',
     'new_vent' : 'New Ventilated Admissions',
     'cen_hosp' : 'Hospitalized Census',
     'cen_icu' : 'ICU Census',
     'cen_vent' : 'Ventilated Census',
     'SOURCE_ID': "Source Id",
     'susceptble': 'Susceptible',
     'infected': 'Infected',
     'recovered': 'Recovered'}

    outputAlias = []

    for field in fieldNames:
        if field in alias:
            outputAlias.append(alias[field])
        else:
            outputAlias.append(field)

    return outputAlias

def messageInfo(msg, indexData, indexDate, arraysSum):
    """ Print Message
    INPUT:
        msg (str): String infoDate
        indexData(int): index array with data
        indexDate(int): index array with dates
        arraysSum (list 1d Arrays): arras
    """
    maxValue = arraysSum[indexData].max()
    id = NUM.where(arraysSum[indexData]==maxValue)[0]
    if type(id) == np.ndarray:
        if len(id) > 0:
            id = id[0]
        else:
            return
    date = arraysSum[indexDate][id]
    value = utils.formatValue(maxValue, "%0.1f")
    infoDate = date.astype(datetime.datetime).strftime('%x')
    ARCPY.AddMessage(msg.format(value, infoDate, id))

def generateSummaryMessage(arraysSum):
    """ Generate Summary Message
    INPUT:
        arraysSum (List 1d arrays): Information aggregated
    """
    messageInfo("Hospitalized Census peaks at {0} on {1} (Day {2})", 5,1,arraysSum)
    messageInfo("ICU Census peaks at {0} on {1} (Day {2})", 6,1,arraysSum)
    messageInfo("Ventilated Census peaks at {0} on {1} (Day {2})",7 ,1,arraysSum)
    messageInfo("New Daily Hospitalizations peaks at {0} on {1} (Day {2})",2 ,1,arraysSum)
    messageInfo("New Daily ICU Admissions peaks at {0} on {1} (Day {2})",3 ,1,arraysSum)
    messageInfo("New Daily Ventilated Admissions at {0} on {1} (Day {2})", 4,1,arraysSum)



constant_str = "Set as Constant"
field_str = "Set as Field"
dummy_date = "1/1/1900"
defaultVals = [None] * 17
constantDefaults = [4, 30, 2.5, 0.75, 0.5, 7, 9, 10, 100, 14]
_ = [defaultVals.append(x) for x in constantDefaults]

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "covid19"
        self.alias = "covid19"

        # List of tool classes associated with this toolbox
        self.tools = [CHIME]

    
class CHIME(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "CHIME Model v1.1.2"
        self.description = "Generate time enabled point feature class showing hospital bed demand over time using the CHIME model"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions."""

        param0 = arcpy.Parameter(
            displayName = "Input Feature Class",
            name = "in_fc",
            datatype = "GPFeatureLayer",
            parameterType = "Required",
            direction = "Input")
        param0.filter.list = ['Point', 'Polygon']
        param0.displayOrder = 0

        param1 = arcpy.Parameter(
            displayName = "Daily Forecast Output Feature Class",
            name = "out_fc",
            datatype = "DEFeatureClass",
            parameterType = "Required",
            direction = "Output")
        param1.displayOrder = 1

        param2 = arcpy.Parameter(
            displayName = "Population",
            name = "pop_field",
            datatype = "Field",
            parameterType = "Required",
            direction = "Input")
        param2.parameterDependencies = [param0.name]
        param2.filter.list = ['Short', 'Long', 'Double', 'Float']
        param2.displayOrder = 2

        param3 = arcpy.Parameter(
            displayName="Number of Currently Known Infections",
            name="known_infections",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param3.parameterDependencies = [param0.name]
        param3.displayOrder = 27
        param3.category = 'Additional Outputs for Visualization'
        param3.filter.list = ['Short', 'Long', 'Double', 'Float']
        param3.enabled = False

        param4 = arcpy.Parameter(
            displayName="Number of Currently Hospitalized COVID-19 Patients",
            name="hosp_field",
            datatype="Field",
            parameterType="Required",
            direction="Input")
        param4.filter.list = ['Short', 'Long', 'Double', 'Float']
        param4.parameterDependencies = [param0.name]
        param4.displayOrder = 3

        ## Field-Based Input Parameters
        param5 = arcpy.Parameter(
            displayName="Doubling Time in Days (Up to Today)",
            name="doubling_time_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param5.parameterDependencies = [param0.name]
        param5.filter.list = ['Short', 'Long', 'Double','Float']
        param5.category = 'Field-Based Model Parameters'
        param5.displayOrder = 4

        param6 = arcpy.Parameter(
            displayName="Social Distancing % (Reduction in Social Contact Going Forward)",
            name="social_dist_perc_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param6.parameterDependencies = [param0.name]
        param6.filter.list = ['Short', 'Long', 'Double','Float']
        param6.category = 'Field-Based Model Parameters'
        param6.displayOrder = 5

        param7 = arcpy.Parameter(
            displayName="Hospitalization % (Total Infections)",
            name="hosp_rate_perc_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param7.parameterDependencies = [param0.name]
        param7.filter.list = ['Short', 'Long', 'Double','Float']
        param7.category = 'Field-Based Model Parameters'
        param7.displayOrder = 6

        param8 = arcpy.Parameter(
            displayName="ICU % (Total Infections)",
            name="icu_perc_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param8.parameterDependencies = [param0.name]
        param8.filter.list = ['Short', 'Long', 'Double','Float']
        param8.category = 'Field-Based Model Parameters'
        param8.displayOrder = 7

        param9 = arcpy.Parameter(
            displayName="Ventilated % (Total Infections)",
            name="vent_perc_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param9.parameterDependencies = [param0.name]
        param9.filter.list = ['Short', 'Long', 'Double','Float']
        param9.category = 'Field-Based Model Parameters'
        param9.displayOrder = 8

        param10 = arcpy.Parameter(
            displayName="Average Hospital Length of Stay (Days)",
            name="hosp_stay_len_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param10.parameterDependencies = [param0.name]
        param10.filter.list = ['Short', 'Long', 'Double','Float']
        param10.category = 'Field-Based Model Parameters'
        param10.displayOrder = 10

        param11 = arcpy.Parameter(
            displayName="Average Days in ICU",
            name="icu_stay_len_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param11.parameterDependencies = [param0.name]
        param11.filter.list = ['Short', 'Long', 'Double','Float']
        param11.category = 'Field-Based Model Parameters'
        param11.displayOrder = 11

        param12 = arcpy.Parameter(
            displayName="Average Days on Ventilator",
            name="vent_stay_len_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param12.parameterDependencies = [param0.name]
        param12.filter.list = ['Short', 'Long', 'Double','Float']
        param12.category = 'Field-Based Model Parameters'
        param12.displayOrder = 12

        param13 = arcpy.Parameter(
            displayName="Hospital Market Share %",
            name="hosp_market_share_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param13.parameterDependencies = [param0.name]
        param13.filter.list = ['Short', 'Long', 'Double','Float']
        param13.category = 'Field-Based Model Parameters'
        param13.enabled = False
        param13.displayOrder = 13

        param14 = arcpy.Parameter(
            displayName="Infectious Days",
            name="inf_days_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param14.parameterDependencies = [param0.name]
        param14.filter.list = ['Short', 'Long', 'Double','Float']
        param14.category = 'Field-Based Model Parameters'
        param14.displayOrder = 9

        ## Constant Input Parameters
        param15 = arcpy.Parameter(
            displayName = "Number of Days to Project",
            name = "num_days",
            datatype = "GPLong",
            parameterType = "Required",
            direction = "Input")
        param15.value = 60
        param15.filter.type = "Range"
        param15.filter.list = [30, 365]
        param15.displayOrder = 14

        param16 = arcpy.Parameter(displayName = "Start Date",
                                 name = "start_date",
                                 datatype = 'GPDate',
                                 parameterType = "Required",
                                 direction = "Input")
        dat = date.today()
        param16.value =  dt(dat.year, dat.month, dat.day)
        param16.displayOrder = 15

        param17 = arcpy.Parameter(
            displayName="Doubling Time in Days (Up to Today)",
            name="doubling_time",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param17.value = str(4)
        param17.category = 'Constant Model Parameters'
        param17.displayOrder = 17

        param18 = arcpy.Parameter(
            displayName="Social Distancing % (Reduction in Social Contact Going Forward)",
            name="social_distancing_perc",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param18.value = str(30)
        param18.category = 'Constant Model Parameters'
        param18.displayOrder = 18

        param19 = arcpy.Parameter(
            displayName="Hospitalization % (Total Infections)",
            name="hosp_rate_perc",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param19.value = str(2.5)
        param19.category = 'Constant Model Parameters'
        param19.displayOrder = 19

        param20 = arcpy.Parameter(
            displayName="ICU % (Total Infections)",
            name="icu_perc",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param20.value = str(0.75)
        param20.category = 'Constant Model Parameters'
        param20.displayOrder = 20

        param21 = arcpy.Parameter(
            displayName="Ventilated % (Total Infections)",
            name="vent_perc",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param21.value = str(0.5)
        param21.category = 'Constant Model Parameters'
        param21.displayOrder = 21

        param22 = arcpy.Parameter(
            displayName="Average Hospital Length of Stay (Days)",
            name="hosp_stay_len",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param22.value = str(7)
        param22.category = 'Constant Model Parameters'
        param22.displayOrder = 23

        param23 = arcpy.Parameter(
            displayName="Average Days in ICU",
            name="icu_stay_len",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param23.value = str(9)
        param23.category = 'Constant Model Parameters'
        param23.displayOrder = 24

        param24 = arcpy.Parameter(
            displayName="Average Days on Ventilator",
            name="vent_stay_len",
            datatype="GPString",
            parameterType="Required",
            direction="Input")
        param24.value = str(10)
        param24.category = 'Constant Model Parameters'
        param24.displayOrder = 25

        param25 = arcpy.Parameter(
            displayName="Hospital Market Share %",
            name="hosp_market_share",
            datatype="GPString",
            parameterType="Optional",
            direction="Input")
        param25.value = str(100)
        param25.category = 'Constant Model Parameters'
        param25.enabled = False
        param25.displayOrder = 26

        param26 = arcpy.Parameter(
            displayName="Infectious Days",
            name="inf_days",
            datatype="GPString",
            parameterType="Optional",
            direction="Input")
        param26.parameterDependencies = [param0.name]
        param26.value = str(14)
        param26.category = 'Constant Model Parameters'
        param26.displayOrder = 22

        ## Additional Field Outputs
        param27 = arcpy.Parameter(
            displayName="Total Bed Capacity",
            name="lic_bed_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param27.parameterDependencies = [param0.name]
        param27.filter.list = ['Short', 'Long', 'Double', 'Float']
        param27.displayOrder = 28
        param27.category = 'Additional Outputs for Visualization'
        param27.enabled = True

        param28 = arcpy.Parameter(
            displayName="Total Ventilator Capacity",
            name="staffed_vent_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param28.parameterDependencies = [param0.name]
        param28.filter.list = ['Short', 'Long', 'Double', 'Float']
        param28.displayOrder = 29
        param28.category = 'Additional Outputs for Visualization'
        param28.enabled = True

        param29 = arcpy.Parameter(
            displayName="Total ICU Bed Capacity",
            name="icu_beds_field",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param29.parameterDependencies = [param0.name]
        param29.filter.list = ['Short', 'Long', 'Double', 'Float']
        param29.displayOrder = 30
        param29.category = 'Additional Outputs for Visualization'
        param29.enabled = True

        param30 = arcpy.Parameter(displayName="Additional Output Variable(s)",
                            name = "additional_fields",
                            datatype = "Field",
                            parameterType = "Optional",
                            direction = "Input",
                            multiValue = True)
        param30.parameterDependencies = [param0.name]
        param30.displayOrder = 31
        param30.category = 'Additional Outputs for Visualization'
        param30.enabled = True

        param31 = arcpy.Parameter(
            displayName="Unique ID",
            name="unique_id",
            datatype="Field",
            parameterType="Optional",
            direction="Input")
        param31.parameterDependencies = [param0.name]
        param31.filter.list = ['Long', 'Text']
        param31.displayOrder = 16

        param32 = arcpy.Parameter(
            displayName = "Summary Output Feature Class",
            name = "out_fc2d",
            datatype = "DEFeatureClass",
            parameterType = "Optional",
            direction = "Output")
        param32.displayOrder = 1
        param32.enabled = True

        parameters = [param0, param1, param2, param3, param4, param5, param6, param7,
                     param8, param9, param10, param11, param12, param13, param14, 
                     param15, param16, param17, param18, param19, param20, param21,
                     param22, param23, param24, param25, param26, param27, param28,
                     param29, param30, param31, param32]

        return parameters

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        # try:
        #     if arcpy.CheckExtension("Spatial") == "Available":
        #         arcpy.CheckOutExtension("Spatial")
        #     else:
        #         raise Exception
        # except:
        #     return False
        return True

    def updateParameters(self,parameters) :
        """Modify the values and properties of parameters before internal validation is
        performed. This method is called whenever a parameter has been changed."""
        self.parameters = parameters
        num_param = 10
        jump_param = 12
        start_param = 5

        for ind in range(num_param):
            if self.parameters[start_param + ind].altered:
                if self.parameters[start_param + ind].value:
                    self.parameters[start_param + jump_param + ind].value = field_str
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool parameter.
        This method is called after internal validation."""
        # check doubling date, should be no less than 0.5
        dbl_ind = 17
        ind_lag = 12
        checkValue(parameters[dbl_ind], minValue= 0.5, onlyCheckLowerBound = True, 
                   defaultVal = defaultVals[dbl_ind], fieldParameter = parameters[dbl_ind - ind_lag])
        
        ## Check Day Parameters >= 1
        minValue = 1
        for i in [22, 23, 24, 26]:
            checkValue(parameters[i], defaultVal = defaultVals[i],  minValue = minValue, onlyCheckLowerBound = True,
                       fieldParameter = parameters[i - ind_lag])

        for i in [18, 20, 21, 25]:
            checkValue(parameters[i], defaultVal = defaultVals[i], fieldParameter = parameters[i - ind_lag])

        ## Check Percentage Parameters >0
        minValue = 0
        for i in [19]:
            checkValue(parameters[i], defaultVal = defaultVals[i],  minValue=minValue, onlyCheckLowerBound = True,
                       fieldParameter = parameters[i - ind_lag], minExclude = True)

        for i, j in [(2,3),(3,4),(2,4)]:
            if parameters[i].value and parameters[j].value:
                if parameters[i].value.value == parameters[j].value.value:
                    parameters[j].setErrorMessage(fr"The field {parameters[j].value.value} is used in the parameter {parameters[i].displayName}")

        return

    # -----**************************************************************-----#

    def execute(self, parameters, messages):
        '''
        CHIME Model Execution
        '''
        import pdb
        proj = arcpy.GetInstallInfo()
        version = proj['Version'].split('.')
        if version[0] == '2':
            v2x = int(version[1])
            if v2x < 3:
                ARCPY.AddError("This tool requires ArcGIS Pro 2.3 or later")
                raise SystemExit()
        else:
            ARCPY.AddError("This tool requires ArcGIS Pro 2.3 or later")
            raise SystemExit()
            
        path = os.path.abspath(__file__)

        from warnings import simplefilter
        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)

        in_features = utils.getTextParameter(0, parameters)
        output_fc = utils.getTextParameter(1, parameters)
        output_fc2d = utils.getTextParameter(32, parameters)
        allInFields = [utils.getTextParameter(x, parameters) for x in range(2, 15)]
        allInConsts = [utils.getTextParameter(x, parameters) for x in range(15, 27)]
        allOutFields = [utils.getTextParameter(x, parameters) for x in range(27, 30)]
        unique_id = utils.getTextParameter(31, parameters) 
        outStr = utils.getTextParameter(30, parameters)
        if outStr:
            allOutFields += outStr.split(";")
        duration = utils.getNumericParameter(15, parameters)
        start_time = parameters[16].value

        dictOpt = {"hos":utils.getTextParameter(27, parameters),
                   "icu":utils.getTextParameter(29, parameters), 
                   "vnt":utils.getTextParameter(28, parameters)}

        ## Set Hospitalziation % to 100
        allInConsts[10] = 100
        ## Wrangle Input Fields
        origFields = [ f for f in allOutFields if f]

        allOutFields = [f.upper() for f in allOutFields if f]
        # Get Fields for SSDO
        allFields = allInFields + allOutFields
        readFields = [f.upper() for f in allFields if f]

        readFields = list(np.unique(readFields)) 

        if unique_id not in ["", None]:
            readFields.append(unique_id.upper())

        # Create a Spatial Stats Data Object (SSDO) ####
        ssdo = SSDO.SSDataObject(in_features, templateFC = output_fc,
                                 useChordal = False)

        isTable = False
        getGeometry = False
        layer = os.path.join(os.path.dirname(path), "CHIME_graduated_symbols_POINT_output_census.lyrx")
        if ssdo.shapeType.upper() != "POINT":
            getGeometry = True
            layer = os.path.join(os.path.dirname(path), "CHIME_graduated_symbols_POLY_output_census.lyrx")

        try:
            parameters[1].symbology = layer
        except:
            ARCPY.AddIDMessage("WARNING", 973)
            pass

        ssdo.obtainData(ssdo.oidName, readFields, minNumObs = 1, 
                        requireGeometry = getGeometry)

        origFieldsType = [ utils.convertType[ssdo.fields[c.upper()].type] for c in origFields  if c ]
        origFieldsDtype = {c.upper():ssdo.fields[c.upper()].data.dtype for c in origFields  if c }

        #### Get Id ####
        uniqueValue = None
        typeUnique = "TEXT"

        if unique_id not in ["", None]:
            npType = ssdo.fields[unique_id.upper()].data.dtype
            uniqueValue = ssdo.fields[unique_id.upper()].data
            if uniqueValue.dtype == NUM.int32:
                typeUniqe = "LONG"

        fieldNames = ['DblTime', 'SocDistPer', 'InfHospPer', 'InfICUPer', 'InfVentPer',
                      'HospStay', 'ICUStay', 'VentStay', 'HospMrShrP', 'InfDays']

        fieldNamesType = [ "DOUBLE" ]*len(fieldNames)
        data_mat = setInputMatrix(ssdo, allInFields[3:], allInConsts[2:], fieldNames)

        data_mat["Population"] = ssdo.fields[allInFields[0].upper()].data
        data_mat["NumHosp"] = ssdo.fields[allInFields[2].upper()].data

        diff = data_mat["Population"] - data_mat["NumHosp"]
        if (diff <= 0).sum() >=1:
            ids = NUM.where(diff<=0)[0]
            ARCPY.AddError("There are locations where the population is not greater than the number of people hospitalized ((Only first 30 included):" + ", ".join([str(ssdo.order2Master[i]) for id,i in enumerate(ids) if id<30 ]))
            raise SystemExit

        listOther = ["Population", "NumHosp"]

        # Epsilon Zeros for NUmber Hospitalizeds
        data_mat.loc[data_mat["NumHosp"] == 0, "NumHosp"] = 1e-06

        others = [ssdo.fields[allInFields[0].upper()].data, ssdo.fields[allInFields[2].upper()].data]
        othersType = [utils.convertType[ssdo.fields[allInFields[0].upper()].type], utils.convertType[ssdo.fields[allInFields[2].upper()].type]]
        if allInFields[1]:
            data_mat["NumInf"] = ssdo.fields[allInFields[1].upper()].data
            listOther = ["Population", "NumHosp","NumInf"]

            othersType = [utils.convertType[ssdo.fields[allInFields[0].upper()].type],
                          utils.convertType[ssdo.fields[allInFields[2].upper()].type],
                          utils.convertType[ssdo.fields[allInFields[1].upper()].type]]
        ## PARAMETER CHECKS
        if (data_mat['DblTime'] > duration).any():
            arcpy.AddMessage('Doubling time cannot be greater than model duration')
            raise SystemExit()
        elif (data_mat['DblTime'] < 0.5).any():
            arcpy.AddMessage('Doubling time cannot be less than 0.5')
            raise SystemExit()
        if ((data_mat['SocDistPer'] < 0).any() or (data_mat['SocDistPer'] > 100).any()):
            arcpy.AddMessage('Social distancing % (Reduction in Social Contact Going Forward) must be between 0 and 100')
            raise SystemExit()
        if (data_mat['InfHospPer'] <= 0).any() or (data_mat['InfHospPer'] > 100).any():
            arcpy.AddMessage('Hospitalization % (Total Infections) must be greater than 0 and cannot be more than 100')
            raise SystemExit()
        if ((data_mat[ 'InfICUPer'] < 0).any() or (data_mat['InfICUPer'] > 100).any()):
            arcpy.AddMessage('ICU % (Total Infections) must be between 0 and 100')
            raise SystemExit()
        if ((data_mat[ 'InfVentPer'] < 0).any() or (data_mat['InfVentPer'] > 100).any()):
            arcpy.AddMessage('Ventilated % (Total Infections) must be between 0 and 100')
            raise SystemExit()
        if (data_mat['HospStay'] < 1 ).any():
            arcpy.AddMessage('Average Hospital Length of Stay (days) must be greater than or equal to 1')
            raise SystemExit()
        if (data_mat['ICUStay'] < 1 ).any():
            arcpy.AddMessage('Average days in ICU must be greater than or equal to 1')
            raise SystemExit()
        if (data_mat['VentStay'] < 1 ).any():
            arcpy.AddMessage('Average Days on Ventilator must be greater than or equal to 1')
            raise SystemExit()
        if (data_mat['InfDays'] < 1 ).any():
            arcpy.AddMessage('Infectious days must be greater than or equal to 1')
            raise SystemExit()

        # Get Input Extent and Spatial Reference
        arcpy.env.overwriteOutput = True
        desc = arcpy.Describe(in_features)
        # Set Output Extent and Spatial Reference
        outSpatRef = setEnvSpatialReference(desc.spatialReference)
        spatialExtent = setEnvExtent(desc.extent)
        inputFields = getInputFields(allFields)
        outputFields = ["day", "date", "admits_hospitalized", "admits_icu", "admits_ventilated"]
        outputFieldsShort = ["day", "date", "new_hosp", "new_icu", "new_vent"]
        typeOutputFields = ["DOUBLE", "DATE", "DOUBLE", "DOUBLE", "DOUBLE"]
        outputFieldsCensus = ['census_hospitalized', 'census_icu', 'census_ventilated']
        outputFieldsCensusShort = ["cen_hosp", "cen_icu", "cen_vent"]
        typeOutputFieldsCensus = [ "DOUBLE", "DOUBLE", "DOUBLE"]
        outputFieldsInfect = ["susceptble", "infected", "recovered"]
        typeOutputFieldsInfect = [ "DOUBLE", "DOUBLE", "DOUBLE"]
        fieldsNameOutput =  outputFieldsShort + outputFieldsCensusShort+ outputFieldsInfect
        fieldTypeOutput  =  typeOutputFields + typeOutputFieldsCensus+ typeOutputFieldsInfect
        nOutput = len(fieldTypeOutput)
        nLoc = len(data_mat)

        fieldsNameOutput2d = [r.replace("%","Perc") for r in fieldNames]+ listOther +  [c.upper() for c in origFields]
        fieldTypeOutput2d  = ["DOUBLE"]*len(fieldNames) + othersType +  origFieldsType


        arrays = []
        arraysSum = []
        nl = nLoc*duration
        addArrays(arrays, nl, fieldTypeOutput, origFieldsDtype, fieldsNameOutput)
        addArrays(arraysSum, duration, fieldTypeOutput, origFieldsDtype, fieldsNameOutput)
        arrays2d = []
        addArrays(arrays2d, nLoc, fieldTypeOutput2d, origFieldsDtype, fieldsNameOutput2d)

        if uniqueValue is not None:
            fieldsNameOutput.append(unique_id)
            fieldsNameOutput2d.append(unique_id)
            arrays.append(NUM.empty(nl, npType))
            arrays2d.append(NUM.empty(nLoc, npType))

        #### Add Statistics ####
        fieldNamesStats, fieldTypeStats, fielAliasStats = typeNewFieldsBasicStats()
        addArrays(arrays2d, nLoc, fieldTypeStats, None, None)
        fieldsNameOutput2d.extend(fieldNamesStats)

        #### Add Capacity Arrays ####
        abrvs = []
        for typeD in dictOpt:
            if dictOpt[typeD] is not None:
                abrvs.append(typeD)
        overCapFieldNames, overCapFieldTypes, overCapFieldAlias = typeNewFieldsBasicStatsOver(abrvs)
        addArrays(arrays2d, nLoc, overCapFieldTypes, None, None)
        fieldsNameOutput2d.extend(overCapFieldNames)

        arrays.append(NUM.empty(nl, NUM.int32))
        arrays2d.append(NUM.empty(nLoc, NUM.int32))

        ARCPY.SetProgressor("default","Running CHIME model...")

        unq, cont = NUM.unique([c.upper() for c in  fieldsNameOutput2d+["SOURCE_ID"]], return_counts = True)
        if (cont>1).sum()>0:
            ids = unq[cont>1]
            ARCPY.AddError("There are repeated fields {}.".format(",".join(list(ids))))
            raise SystemExit

        ## CHIME Model
        badLocations = []
        for id, loc in data_mat.iterrows():
            chime_param = Parameters(population = int(loc["Population"]),
                                     doubling_time = loc["DblTime"],
                                     relative_contact_rate = loc["SocDistPer"]/100,
                                     current_hospitalized = loc["NumHosp"],
                                     hospitalized=RateDays(loc['InfHospPer']/100, int(loc['HospStay'])),
                                     icu=RateDays(loc['InfICUPer']/100, int(loc['ICUStay'])),
                                     ventilated=RateDays(loc['InfVentPer']/100, int(loc['VentStay'])),
                                     infectious_days = int(loc["InfDays"]),
                                     date_first_hospitalized = None,
                                     n_days= duration,
                                     current_date=start_time)

            model = Model(chime_param)

            mask = (model.admits_df["day"] >= 0) & (model.admits_df["day"] < duration)
            maskV = mask.values.copy()
            
            df = {}
            for i in  [ 'day', 'date', 'admits_hospitalized', 'admits_icu', 'admits_ventilated']:
                val = model.admits_df[i].values.copy()
                df[i] = val[maskV]


            n = len(loc)
            df_census = {}

            for i in  [ 'census_hospitalized', 'census_icu', 'census_ventilated']:
                val = model.census_df[i].values.copy()
                df_census[i] = val[maskV]
                
            df_census['day']= df['day']
            df_census['date']= df['date']

            df_infect = {}
            for i in  [ 'susceptible', 'infected', 'recovered']:
                val = model.sim_sir_w_date_df[i].values.copy()
                if i == 'susceptible':
                    i = 'susceptble'
                df_infect[i] = val[maskV]

            values = loc.values

            index2d =0
            for index2d in NUM.arange(n):
                arrays2d[index2d][id] = values[index2d]

            #### 3D Input #################################################3
            slc = slice(int(id*duration),int((id+1)*duration))

            index = 0
            index = updateArray(arrays, index, slc, df, outputFields, typeOutputFields, arraysSum )
            index = updateArray(arrays, index, slc, df_census, outputFieldsCensus, typeOutputFieldsCensus, arraysSum)
            index = updateArray(arrays, index, slc, df_infect, outputFieldsInfect, typeOutputFieldsInfect, arraysSum)

            if uniqueValue is not None:
                arrays[index][slc] =  NUM.array([uniqueValue[id]]*duration, npType )
                index += 1

            arrays[index][slc] = NUM.array([ssdo.order2Master[id]]*duration, NUM.int32 )

            #### End 3D  ####################################################

            if len(origFieldsType)>0:
                for name in origFields :
                    index2d+=1
                    arrays2d[index2d][id] = ssdo.fields[name.upper()].data[id]

            if uniqueValue is not None:
                index2d+=1
                arrays2d[index2d][id] =  uniqueValue[id]

            #### Add Statistics ####
            information = basicStats(df_census, nameH="census_hospitalized", nameI = "census_icu", nameV = "census_ventilated")
            if information[0] is None:
                badLocations.append(str(ssdo.order2Master[id]))
                continue

            for valStat in information:
                index2d+=1
                arrays2d[index2d][id] =  valStat

            if dictOpt["hos"]is not None:
                information = overCapStats(df_census, id, ssdo, dictOpt["hos"], 'census_hospitalized', duration)
                for valStat in information:
                    index2d+=1
                    arrays2d[index2d][id] =  valStat

            if dictOpt["icu"] is not None:
                information = overCapStats(df_census, id, ssdo, dictOpt["icu"], 'census_icu', duration)

                for valStat in information:
                    index2d+=1
                    arrays2d[index2d][id] =  valStat

            if dictOpt["vnt"] is not None:
                information = overCapStats(df_census, id, ssdo, dictOpt["vnt"], 'census_ventilated', duration)
                for valStat in information:
                    index2d+=1
                    arrays2d[index2d][id] =  valStat

            arrays2d[index2d+1][id] = ssdo.order2Master[id]

        if len(badLocations) > 0 and len(badLocations) != nLoc:
            loc = [str(l) for idd, l in enumerate(badLications) if idd < 30]
            infobLoc = ", ".join(loc)
            ARCPY.AddWarning(fr"It is not possible to use the model in the location(s) (first 30 locations) {infobLoc}")

        if len(badLocations) == nLoc:
            ARCPY.AddError(fr"CHIME found problems in the calculation in all locations")
            raise SystemExit

        #### Create Summary ####
        generateSummaryMessage(arraysSum)

        #### Select Method to Create Output ####
        VersionContainer = ContainerVersion()

        if getGeometry:

            shapes = []
            #### Copy Shapes ####
            for shp in list(ssdo.shapes):
                wkb =  shp.WKB
                for i in NUM.arange(duration):
                    shapes.append(arcpy.FromWKB(wkb))

            zValues = None
            if ssdo.hasZ:
                zValues = ssdo.zCoords

            container = VersionContainer(spatialRef = ssdo.spatialRef, shapes = shapes, z = zValues)
            fields3d = fieldsNameOutput + ["SOURCE_ID"]
            fields3dAliases = getAliasFields(fields3d)

            container.generateOutput(output_fc, arrays, fields3d, fields3dAliases)
        else:
            n = len(ssdo.xyCoords)
            zValues = None
            if ssdo.hasZ:
                zValues = np.repeat(ssdo.zCoords,duration).ravel()

            container = VersionContainer(spatialRef = ssdo.spatialRef, xy = np.tile(ssdo.xyCoords, duration).reshape(n*duration,2), z = zValues)
            fields3d = fieldsNameOutput + ["SOURCE_ID"]
            fields3dAliases = getAliasFields(fields3d)
            container.generateOutput(output_fc, arrays, fields3d, fields3dAliases)

        chart1 = arcpy.Chart("New Daily Admissions")
        chart1.title = 'New Daily Admissions Projections'
        chart1.type = "line"
        chart1.xAxis.field = "day"
        chart1.line.aggregation = "SUM"
        chart1.yAxis.field = ["new_hosp", "new_icu", "new_vent"]
        chart1.xAxis.title = 'Number of Days from Start'
        chart1.yAxis.title = 'Admissions'
        chart1.color = ["#00A9E6", "#FFAA00", "#ED7551"]

        chart2 = arcpy.Chart("Daily Hospital Census Projections")
        chart2.title = 'Daily Hospital Census Projections'
        chart2.type= "line"
        chart2.xAxis.field = "day"
        chart2.line.aggregation= "SUM"
        chart2.yAxis.field = ["cen_hosp", "cen_icu", "cen_vent"]
        chart2.xAxis.title = 'Number of Days from Start'
        chart2.yAxis.title = 'Census'
        chart2.color = ["#00A9E6", "#FFAA00", "#ED7551"]

        chart3 = arcpy.Chart("Susceptible, Infected, and Recovered Projections")
        chart3.title = 'Susceptible, Infected, and Recovered Projections'
        chart3.type = "line"
        chart3.xAxis.field = "day"
        chart3.line.aggregation= "SUM"
        chart3.yAxis.field = ["susceptble", "infected", "recovered"]
        chart3.xAxis.title = 'Number of Days from Start'
        chart3.yAxis.title = 'Count'
        chart3.color = ["#B2DF8A", "#F57AB6", "#66CDAB"]

        parameters[1].charts = [chart1, chart2, chart3]

        if output_fc2d not in ["",None]:

            layer2d = os.path.join(os.path.dirname(path), "CHIME_graduated_symbols_POINT_output_SUMMARY.lyrx")
            if ssdo.shapeType.upper() != "POINT":
                layer2d = os.path.join(os.path.dirname(path), "CHIME_graduated_symbols_POLY_output_SUMMARY.lyrx")
            try:
                parameters[32].symbology = layer2d
            except:
                ARCPY.AddIDMessage("WARNING", 973)
                pass

            if getGeometry:
                zValues = None
                if ssdo.hasZ:
                    zValues = ssdo.zCoords
                container = VersionContainer(spatialRef = ssdo.spatialRef, shapes = list(ssdo.shapes), z = zValues)
                fields2d = fieldsNameOutput2d + ["SOURCE_ID"]
                fields2dAliases = getAliasFields(fields2d)
                container.generateOutput(output_fc2d, arrays2d, fields2d, fields2dAliases )

            else:
                n = len(ssdo.xyCoords)
                zValues = None

                if ssdo.hasZ:
                    zValues =ssdo.zCoords

                container = VersionContainer(spatialRef = ssdo.spatialRef, xy = ssdo.xyCoords,z = zValues)

                fields2d = fieldsNameOutput2d + ["SOURCE_ID"]
                fields2dAliases = getAliasFields(fields2d)
                container.generateOutput(output_fc2d, arrays2d,  fields2d, fields2dAliases)


        return
