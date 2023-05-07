'''
Convert intracellular electrophysiology recording data to the NWB format
in Python

Run this script to convert intracellular electrophysiology recording data
and associated optogenetic stimulation data generated at the University
of Bristol (UoB) to the Neurodata Without Borders (NWB) file format. This
script is explained in the accompanying Bristol GIN for Patch Clamp Data
tutorial available at
https://dervinism.github.io/bristol-neuroscience-data-guide/tutorials/Bristol%20GIN%20for%20Patch%20Clamp%20Data.html

You can use this script to get an idea of how to convert your own
intracellular electrophysiology data to the NWB file format.
'''

import scipy.io
import cv2
import numpy as np
from datetime import datetime

from pynwb import NWBFile, NWBHDF5IO
from pynwb.core import DynamicTable, VectorData
from pynwb.file import Subject
from pynwb.base import Images, TimeSeries
from pynwb.image import GrayscaleImage

from localFunctions import getRuns, setVClampSeries, setCClampSeries

# Record metadata
# Project (experiment) metadata:
projectName = 'Inhibitory plasticity experiment in CA1'
experimenter = 'MU'
institution = 'University of Bristol'
publications = 'In preparation'
lab = 'Jack Mellor lab'
brainArea = 'Hippocampus CA1'

# Animal metadata
animalID = '180126'
ageInDays = 34
age = 'P'+str(ageInDays)+'D' # Convert to ISO8601 format: https://en.wikipedia.org/wiki/ISO_8601#Durations
strain = 'Ai32/PVcre'
sex = 'F'
species = 'Mus musculus'
weight = []
description = '001' # Animal testing order.

# Session metadata
startYear = 2018
startMonth = 1
startDay = 26
startTime = datetime(startYear, startMonth, startDay)
year = str(startYear); year = year[2:4]
month = str(startMonth)
if len(month) == 1:
  month = '0'+month
day = str(startDay)
if len(day) == 1:
  day = '0'+day
sliceNumber = 1
cellNumber = 1
sessionID = year + month + day + '__s' + str(sliceNumber) + 'c' + str(cellNumber) # mouse-id_time_slice-id_cell-id
sessionDescription = 'Current and voltage clamp recordings using electric/optogenetic stimulation plasticity-inducing protocol.'
expDescription = 'Optogenetic and current stim pathways were stimulated in an interleaved fashion with a 5 second interval.' +\
                 'Each stimulation pathway consisted of 2 stimulations at 50ms interval: 2 action potentials or 2 light pulses.' +\
                 'After stable baselines in both pathways, plasticity protocol was induced.' +\
                 'After plasticty protocol induced, optogenetic and current stimulation resumed as before.'
sessionNotes = '180126 PV mouse' +\
               'Gender: female' +\
               'DOB: 23/12/17 – 4/5wo' +\
               'genotype: ??' +\
               'ID: 065321 l0 r1' +\
               'in NBQX and AP5' +\
               'NEW  protocol using soph''s' +\
               '0ms gap single pre 4 post spikes with 0ms interval between the pre and 1st post' +\
               'Slice 1' +\
               'Cell1' +\
               'Ok cell died within around 20 mins'

# Assign NWB file fields
nwb = NWBFile(
  session_description = sessionDescription,
  identifier = sessionID, 
  session_start_time = startTime, 
  experimenter = experimenter,  # optional
  session_id = sessionID,  # optional
  institution = institution,  # optional
  related_publications = publications,  # optional
  notes = sessionNotes,  # optional
  lab = lab,  # optional
  experiment_description = expDescription) # optional

# Create subject object
nwb.subject = Subject(
  subject_id = animalID,
  age = age,
  description = description,
  species = species,
  sex = sex)

# Load data
data = scipy.io.loadmat('../' + year + month + day + '__s' + str(sliceNumber) + \
  'c' + str(cellNumber) + '_001_ED.mat', squeeze_me=True)['V' + year + month + day + \
  '__s' + str(sliceNumber) + 'c' + str(cellNumber) + '_001_wave_data']
values = data['values'].all()
vcScaleFactor = 1/10E12
ccScaleFactor = 2.5/10E5

# Extract sweep and run data
sweepIDs = np.int64(data['frameinfo'].all()['number'])
sweepDataPoints = np.int64(data['frameinfo'].all()['points'])
sweepStartTimes = np.double(data['frameinfo'].all()['start'])
sweepStates = np.int64(data['frameinfo'].all()['state'])
sweepLabels = data['frameinfo'].all()['label'].astype('U')
(runs, runInds, runStartTimes, runDataPoints, runUnits) = getRuns(sweepLabels, sweepDataPoints, sweepStartTimes)
nSweeps = sweepIDs.size
nRuns = len(runs)
endRunInds = np.concatenate((runInds[1:], np.array([nSweeps])), axis=0) - 1
runInds = np.concatenate((runInds.reshape(-1,1), endRunInds.reshape(-1,1)), axis=1)

# Convert intracellular electrophysiology data
# Create the recording device object
device = nwb.create_device(
  name = 'Amplifier_Multiclamp_700A',
  description = 'Amplifier for recording intracellular data.',
  manufacturer = 'Molecular Devices')

electrode = nwb.create_icephys_electrode(
  name = 'icephys_electrode',
  description = 'A patch clamp electrode',
  location = 'Cell soma in CA1 of hippocampus',
  slice = 'slice #' + str(sliceNumber),
  device = device)

# Add current and voltage clamp data
stimulusObjects = list()
responseObjects = list()
for sweep in range(nSweeps):
  run = np.where(np.logical_and(runInds[:,0] <= sweep, sweep <= runInds[:,1]))[0][0]
  input = {
    'samplingRate': 1/data['interval'],
    'startTime': sweepStartTimes[sweep],
    'data': values[:,sweep],
    'electrode': electrode,
    'condition': runs[run],
    'stimState': sweepStates[sweep],
    'unit': runUnits[run],
    'sweepOrder': [sweep+1, sweepIDs[sweep]]}
  if runs[run] == 'plasticity':
    input.pop('condition')
    input.pop('stimState')
    input['data'] = input['data']*ccScaleFactor
    (stimulusObj, responseObj) = setCClampSeries(input)
  else:
    input['data'] = input['data']*vcScaleFactor
    (stimulusObj, responseObj) = setVClampSeries(input)
  stimulusObjects.append(stimulusObj)
  responseObjects.append(responseObj)

# Create intracellular recordings table
rowIndices = list()
for sweep in range(nSweeps):
  rowIndices.append(nwb.add_intracellular_recording(
    electrode=electrode,
    stimulus=stimulusObjects[sweep],
    stimulus_start_index=-1,
    stimulus_index_count=-1,
    response=responseObjects[sweep],
    response_start_index=0,
    response_index_count=sweepDataPoints[sweep],
    id=sweepIDs[sweep]))

# Add the sweep metadata category to the intracellular recording table
orderCol = VectorData(
  name='order',
  data=sweepIDs,
  description='Recorded sweep order.')

pointsCol = VectorData(
  name='points',
  data=sweepDataPoints,
  description='The number of data points within the sweep.')

startCol = VectorData(
  name='start',
  data=sweepStartTimes,
  description='The sweep recording start time in seconds.')

stateCol = VectorData(
  name='state',
  data=sweepStates,
  description='The experimental state ID: ' + \
              '0 - light stimulation during the baseline condition.' + \
              '1 - current stimulation during the baseline condition.' + \
              '2 - inhibitory synaptic plasticity induction condition.' + \
              '9 - break between baseline and plasticity induction conditions.')

labelCol = VectorData(
  name='label',
  data=sweepLabels,
  description='The experimental state label.')

conditions = VectorData(
  name='condition',
  data=sweepLabels,
  description='The experimental condition.')

sweepMetadata = DynamicTable(
  name='sweeps',
  description='Sweep metadata.',
  colnames=['order','points','start','state','label','condition'],
  columns=[orderCol,pointsCol,startCol,stateCol,labelCol,conditions])
nwb.intracellular_recordings.add_category(category=sweepMetadata)


# Group sweep references in tables of increasing hierarchy
# Group simultaneous recordings
# Create a simultaneous recordings table with a custom column
# 'simultaneous_recording_tag'
icephys_simultaneous_recordings = nwb.get_icephys_simultaneous_recordings()
icephys_simultaneous_recordings.add_column(
  name='simultaneous_recording_tag',
  description='A custom tag for simultaneous_recordings')

rowIndicesSimRec = list()
for iSweep in range(nSweeps):
  rowIndicesSimRec.append(nwb.add_icephys_simultaneous_recording(
    recordings=np.array(rowIndices[iSweep], ndmin=1),
    id=sweepIDs[iSweep],
    simultaneous_recording_tag='noSimultaneousRecs'))

# Group sequential recordings using the same type of stimulus
# Group indices of sequential recordings in a sequential recordings table
seqGroupCount = 0
for iRun in range(nRuns):
  inds = np.arange(runInds[iRun,0],runInds[iRun,-1],1)
  condStates = sweepStates[inds]
  uniqueSweepStates = np.unique(condStates)
  for iState in range(uniqueSweepStates.size):
    if uniqueSweepStates[iState] == 0:
      stimulusType = 'light'
    elif uniqueSweepStates[iState] == 1:
      stimulusType = 'current'
    elif uniqueSweepStates[iState] == 2:
      stimulusType = 'combined'
    elif uniqueSweepStates[iState] == 9:
      stimulusType = 'noStim'
    inds = np.argwhere(condStates == uniqueSweepStates[iState])
    nwb.add_icephys_sequential_recording(
      simultaneous_recordings=[rowIndicesSimRec[int(i)] for i in inds],
      stimulus_type=stimulusType,
      id=seqGroupCount)
    seqGroupCount += 1
    
# Group recordings into runs
# Group indices of individual runs
runInds = list([[1,2], 3, 4, 5, [6,7]])

# Create a repetitions table
for iRun in range(nRuns):
  nwb.add_icephys_repetition(
    sequential_recordings=np.array(runInds[iRun], ndmin=1),
    id=iRun)
  
# Group runs into experimental conditions
# Group indices, tags, and descriptions for different conditions
condInds = list([[1,5], [2,4], 3])
condTags = list(['baselineStim','noStim','plasticityInduction'])

# Create experimental conditions table
for iCond in range(len(condInds)):
  nwb.add_icephys_experimental_condition(
    repetitions=np.array(condInds[iCond], ndmin=1),
    id=iCond)
nwb.icephys_experimental_conditions.add_column(
  name='tag',
  data=condTags,
  description='Experimental condition label')


# Add images
imageFilename = '../' + year + month + day + ' s' + str(sliceNumber) + 'c' + str(cellNumber) + '.jpg'
sliceImage = cv2.imread(imageFilename, cv2.IMREAD_GRAYSCALE)

sliceImage = GrayscaleImage(
  name = 'slice_image',
  data = sliceImage, # required: [height, width]
  description = 'Grayscale image of the recording slice.')

imageCollection = Images(
  name = 'ImageCollection',
  images = [sliceImage],
  description = 'A container for slice images.')

nwb.add_acquisition(imageCollection)


# Save the converted NWB file
with NWBHDF5IO(sessionID + '.nwb', "w") as io:
  io.write(nwb)