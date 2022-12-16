import importlib
import sys
import csv
import datetime
import os
import numpy as np
import math

import htm
from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.date import DateEncoder
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor

def runModel(gymName, fileName, plot=False, load=False):
    # Assumning both plot and load are never passed as true.
    print("Creating model from %s."%gymName)

    default_parameters = {
    # there are 2 (3) encoders: "value" (RDSE) & "time" (DateTime weekend, timeOfDay)
    'enc': {
        "value" :
            {'resolution': 0.88, 'size': 700, 'sparsity': 0.02},
        "time":
            {'timeOfDay': (30, 1), 'weekend': 21}
    },
    'predictor': {'sdrc_alpha': 0.1},
    'sp': {'boostStrength': 2.0,
            'columnCount': 2048,
            'globalInhibition': 1,
            'inputWidht': 0,
            'numActiveColumnsPerInhArea': 40,
            'localAreaDensity': 0.04395604395604396,
            'potentialPct': 0.8,
            'synPermActiveInc': 0.05,
            'synPermConnected': 0.1,
            'seed': 1956,
            'synPermInactiveDec': 0.098130625},
    'tm': {'activationThreshold': 12,
            'cellsPerColumn': 32,
            'globalDecay': 0.0,
            'inputWidth': 2048,
            'maxAge': 0,
            'seed': 1960
            'columnCount':2048,
            'initialPerm': 0.21,
            'maxSegmentsPerCell': 128,
            'maxSynapsesPerSegment': 32,
            'minThreshold': 9,
            'newSynapseCount': 20,
            'permanenceDec': 0.1,
            'permanenceInc': 0.1},
    'anomaly': {'period': 1000},
    }

    # Create encoders
    parameters = default_parameters
    dateEncoder = DateEncoder(timeOfDay= parameters["enc"]["time"]["timeOfDay"],
                            weekend  = parameters["enc"]["time"]["weekend"])
    scalarEncoderParams            = RDSE_Parameters()
    scalarEncoderParams.size       = parameters["enc"]["value"]["size"]
    scalarEncoderParams.sparsity   = parameters["enc"]["value"]["sparsity"]
    scalarEncoderParams.resolution = parameters["enc"]["value"]["resolution"]
    scalarEncoder = RDSE( scalarEncoderParams )
    encodingWidth = (dateEncoder.size + scalarEncoder.size)
    enc_info = Metrics( [encodingWidth], 999999999 )


    # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
    spParams = parameters["sp"]
    sp = SpatialPooler(
        inputDimensions            = (encodingWidth,),
        columnDimensions           = (spParams["columnCount"],),
        potentialPct               = spParams["potentialPct"],
        potentialRadius            = encodingWidth,
        globalInhibition           = True,
        localAreaDensity           = spParams["localAreaDensity"],
        synPermInactiveDec         = spParams["synPermInactiveDec"],
        synPermActiveInc           = spParams["synPermActiveInc"],
        synPermConnected           = spParams["synPermConnected"],
        boostStrength              = spParams["boostStrength"],
        wrapAround                 = True
    )
    sp_info = Metrics( sp.getColumnDimensions(), 999999999 )

    tmParams = parameters["tm"]
    tm = TemporalMemory(
        columnDimensions          = (spParams["columnCount"],),
        cellsPerColumn            = tmParams["cellsPerColumn"],
        activationThreshold       = tmParams["activationThreshold"],
        initialPermanence         = tmParams["initialPerm"],
        connectedPermanence       = spParams["synPermConnected"],
        minThreshold              = tmParams["minThreshold"],
        maxNewSynapseCount        = tmParams["newSynapseCount"],
        permanenceIncrement       = tmParams["permanenceInc"],
        permanenceDecrement       = tmParams["permanenceDec"],
        predictedSegmentDecrement = 0.0,
        maxSegmentsPerCell        = tmParams["maxSegmentsPerCell"],
        maxSynapsesPerSegment     = tmParams["maxSynapsesPerSegment"]
    )
    tm_info = Metrics( [tm.numberOfCells()], 999999999 )

    anomaly_history = AnomalyLikelihood(parameters["anomaly"]["period"])

    predictor = Predictor( steps=[1, 5], alpha=parameters["predictor"]['sdrc_alpha'] )
    predictor_resolution = 1


    # Set file
    records = []
    with open(fileName, "r") as fin:
        reader = csv.reader(fin)
        headers = next(reader)
        next(reader)
        next(reader)
        for record in reader:
            records.append(record)


    # Iterate through every datum in the dataset, record the inputs & outputs.
    inputs      = []
    anomaly     = []
    anomalyProb = []
    predictions = {1: [], 5: []}


    for count, record in enumerate(records):
        # Convert date string into Python date object.
        dateString = datetime.datetime.strptime(record[0], "%Y-%m-%d %H:%M:%S.%f")
        # Convert data value string into float.
        consumption = float(record[2])
        inputs.append( consumption )

        # Call the encoders to create bit representations for each value.  These are SDR objects.
        dateBits        = dateEncoder.encode(dateString)
        consumptionBits = scalarEncoder.encode(consumption)

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( encodingWidth ).concatenate([consumptionBits, dateBits])
        enc_info.addData( encoding )

        # Create an SDR to represent active columns, This will be populated by the
        # compute method below. It must have the same dimensions as the Spatial Pooler.
        activeColumns = SDR( sp.getColumnDimensions() )
        # Execute Spatial Pooling algorithm over input space.
        sp.compute(encoding, True, activeColumns)
        sp_info.addData( activeColumns )

        # Execute Temporal Memory algorithm over active mini-columns.
        tm.compute(activeColumns, learn=True)
        tm_info.addData( tm.getActiveCells().flatten() )

        # Predict what will happen, and then train the predictor based on what just happened.
        pdf = predictor.infer( tm.getActiveCells() )
        for n in (1, 5):
            if pdf[n]:
                predictions[n].append( np.argmax( pdf[n] ) * predictor_resolution )
            else:
                predictions[n].append(float('nan'))

        anomaly.append( tm.anomaly )
        anomalyProb.append( anomaly_history.compute(tm.anomaly) )

        predictor.learn(count, tm.getActiveCells(), int(consumption / predictor_resolution))

    # Print information & statistics about the state of the HTM.
    print("Encoded Input", enc_info)
    print("")
    print("Spatial Pooler Mini-Columns", sp_info)
    print(str(sp))
    print("")
    print("Temporal Memory Cells", tm_info)
    print(str(tm))
    print("")

    # Shift the predictions so that they are aligned with the input they predict.
    for n_steps, pred_list in predictions.items():
        for x in range(n_steps):
            pred_list.insert(0, float('nan'))
            pred_list.pop()

    # Calculate the predictive accuracy, Root-Mean-Squared
    accuracy         = {1: 0, 5: 0}
    accuracy_samples = {1: 0, 5: 0}

    for idx, inp in enumerate(inputs):
        for n in predictions: # For each [N]umber of time steps ahead which was predicted.
            val = predictions[n][ idx ]
            if not math.isnan(val):
                accuracy[n] += (inp - val) ** 2
                accuracy_samples[n] += 1
    for n in sorted(predictions):
        accuracy[n] = (accuracy[n] / accuracy_samples[n]) ** .5
        print("Predictive Error (RMS)", n, "steps ahead:", accuracy[n])

    # Show info about the anomaly (mean & std)
    print("Anomaly Mean", np.mean(anomaly))
    print("Anomaly Std ", np.std(anomaly))

if __name__ == "__main__":
    runModel("common", "disease_person3_converted.csv", False, False)

