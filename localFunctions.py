'''
Local functions for convert2nwbpClamp
'''

import numpy as np
from pynwb.icephys import CurrentClampSeries, CurrentClampStimulusSeries, VoltageClampSeries, VoltageClampStimulusSeries
from typing import TypedDict, Any


def getRuns(sweepLabels: np.ndarray, sweepDataPoints: np.ndarray, sweepStartTimes: np.ndarray) \
-> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
  '''
  [runs, inds, startTimes, dataPoints, units] = getRuns(sweepLabels, sweepDataPoints, sweepStartTimes)
  
  Identifies recording runs and their starting indices and times.

  The function takes in 1D numpy arrays with individual recording sweep information and outputs
  information about runs, their start times, trial durations, etc.

  Args:
    sweepLabels (numpy.ndarray[S]): a shape-(M, 1) array containing sweep labels.
    dataPoints (numpy.ndarray[S]): a shape-(M, 1) array containing datapoint info for each sweep.
    sweepStartTimes (np.ndarray[f]): a shape-(M, 1) array containing sweep start times.
  
  Returns:
    runs (list[str]): names of individual runs.
    inds (np.ndarray[i]): a shape-(N, 1) array containing start sweep indices of individual runs.
    startTimes (np.ndarray[f]): a shape-(N, 1) array containing start times of individual runs.
    dataPoints (np.ndarray[i]): a shape-(N, 1) array containing data points per sweep of individual runs.
    units (list[str]): data measurement units of individual runs.
  '''

  runs = list(['baseline'])
  inds = np.array([0])
  dataPoints = np.array([sweepDataPoints[0]])
  startTimes = np.array([sweepStartTimes[0]])
  sweepUnits = 'amperes'
  units = list([sweepUnits])
  for sweep in range(1,sweepLabels.size-1):
    if not sweepLabels[sweep][0] == sweepLabels[sweep-1][0]:
      if sweepLabels[sweep][0] == 'b':
        runs.append('break')
        units.append('amperes')
      elif sweepLabels[sweep][0] == '0':
        runs.append('plasticity')
        units.append('volts')
      elif sweepLabels[sweep][0] == '1':
        runs.append('baseline')
        units.append('amperes')
      inds = np.append(inds, sweep)
      dataPoints = np.append(dataPoints, sweepDataPoints[sweep])
      startTimes = np.append(startTimes, sweepStartTimes[sweep])

  return runs, inds, startTimes, dataPoints, units


inputStruct = TypedDict('inputStruct', {'samplingRate': float, 'startTime': float, \
                                        'data': np.ndarray, 'electrode': Any, \
                                        'condition': str, 'stimState': int, 'unit': str, \
                                        'sweepOrder': list})
def setVClampSeries(input: inputStruct) -> tuple[Any, Any]:
  '''
  (VCSS, VCS) = setVClampSeries(input)

  Create VoltageClampStimulusSeries and VoltageClampSeries objects.
  
  Function creates, names, and annotates stimulus and response
  equivalents for a voltage clamp given the data. The response
  data is reused when creating the stimulus as no stimulus data
  exists.

  Args:
    input (dict): - a dictionary with the following keys:
      samplingRate (float): a data sampling rate in Hz.
      startTime (float): the time of the first data sample in the
        sweep in seconds
      data (np.ndarray[f]): a shape-(M, N) array containing somatic
        voltage clamp recordings. The first dimension is time and
        the second dimension corresponds to individual sweeps.
      electrode: an electrode object.
      condition (str): the experimental condition or the run name.
      stimState (int): the stimulation state ID.
      unit (str): a data measurement unit.
      sweepOrder (list): the sweep order number list with absolute
        and sweep ID values.

  Returns:
    VCSS: the newly created VoltageClampStimulusSeries object.
    VCS: the newly created VoltageClampSeries object.
  '''

  if input['sweepOrder'][0] < 10:
    prefix = '00'
  elif input['sweepOrder'][0] < 100:
    prefix = '0'
  else:
    prefix = ''
  
  if input['condition'] == 'baseline':
    if input['stimState'] == 0:
      description = 'Baseline condition: Light stimulation'
      stimDescription = 'Baseline stimulation: Double light pulses.'
    elif input['stimState'] == 1:
      description = 'Baseline condition: Current stimulation'
      stimDescription = 'Baseline stimulation: Double current pulses.'
  elif input['condition'] == 'break':
    description = 'Break sweeps are used while switching between two conditions: Nothing happens.'
    stimDescription = 'No stimulation.'
  
  voltageClampStimulusSeries = VoltageClampStimulusSeries(
    name = 'PatchClampSeries' + prefix + str(input['sweepOrder'][0]),
    description = description,
    data = input['data'],
    gain = 1.,
    unit = 'volts',
    electrode = input['electrode'],
    stimulus_description = stimDescription,
    starting_time = input['startTime'],
    rate = input['samplingRate'],
    sweep_number = input['sweepOrder'][1])
  
  voltageClampSeries = VoltageClampSeries(
    name = 'PatchClampSeries' + prefix + str(input['sweepOrder'][0]),
    description = description,
    data = input['data'],
    gain = 1.,
    unit = input['unit'],
    electrode = input['electrode'],
    stimulus_description = stimDescription,
    starting_time = input['startTime'],
    rate = input['samplingRate'],
    sweep_number = input['sweepOrder'][1])
  
  return voltageClampStimulusSeries, voltageClampSeries


inputStruct = TypedDict('inputStruct', {'samplingRate': float, 'startTime': float, \
                                        'data': np.ndarray, 'electrode': Any, \
                                        'unit': str, 'sweepOrder': int})
def setCClampSeries(input: inputStruct) -> tuple[Any, Any]:
  '''
  (CCSS, CCS) = setCClampSeries(input)

  Create CurrentClampStimulusSeries and CurrentClampSeries objects.
  
  Function creates, names, and annotates stimulus and response
  equivalents for a current clamp given the data. The response
  data is reused when creating the stimulus as no stimulus data
  exists.

  Args:
    input (dict): - a dictionary with the following keys:
      samplingRate (float): a data sampling rate in Hz.
      startTime (float): the time of the first data sample in the
        sweep in seconds
      data (np.ndarray[f]): a shape-(M, N) array containing somatic
        current clamp recordings. The first dimension is time and
        the second dimension corresponds to individual sweeps.
      electrode: an electrode object.
      unit (str): a data measurement unit.
      sweepOrder (list): the sweep order number list with absolute
        and sweep ID values.

  Returns:
    CCSS: the newly created CurrentClampStimulusSeries object.
    CCS: the newly created CurrentClampSeries object.
  '''

  if input['sweepOrder'][0] < 10:
    prefix = '00'
  elif input['sweepOrder'][0] < 100:
    prefix = '0'
  else:
    prefix = ''

  currentClampStimulusSeries = CurrentClampStimulusSeries(
    name = 'PatchClampSeries' + prefix + str(input['sweepOrder'][0]),
    description = 'Plasticity condition',
    data = input['data'],
    gain = 1.,
    unit = 'amperes',
    electrode = input['electrode'],
    stimulus_description = 'Plasticity protocol: Simultaneous current and light stimulation',
    starting_time = input['startTime'],
    rate = input['samplingRate'],
    sweep_number = input['sweepOrder'][1])
  
  currentClampSeries = CurrentClampSeries(
    name = 'PatchClampSeries' + prefix + str(input['sweepOrder'][0]),
    description = 'Plasticity condition',
    data = input['data'],
    gain = 1.,
    unit = input['unit'],
    electrode = input['electrode'],
    stimulus_description = 'Plasticity protocol: Simultaneous current and light stimulation',
    starting_time = input['startTime'],
    rate = input['samplingRate'],
    sweep_number = input['sweepOrder'][1])
  
  return currentClampStimulusSeries, currentClampSeries