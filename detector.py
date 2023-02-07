from essentia.standard import FrameGenerator, SNR
from scipy import stats

def detector(trainSet, liveSet, parameters):
    firstColumnLive = liveSet.axes[1][0]
    firstColumnTrain = trainSet.axes[1][0]
    snr_live=[]
    snr_train=[]
    frameSize = 512
    noiseThreshold=-40
    for clip in trainSet[firstColumnTrain]:
        snr = SNR(sampleRate=16000, noiseThreshold=noiseThreshold, useBroadbadNoiseCorrection=False, frameSize = frameSize)
        for frame in FrameGenerator(
            clip,
            frameSize=frameSize,
            hopSize=frameSize // 2,
        ):
            snr_instant, snr_ema, snr_spetral = snr(frame)
        snr_train.append(snr_ema)

    for clip in liveSet[firstColumnLive]:
        snr = SNR(sampleRate=16000, noiseThreshold=noiseThreshold, useBroadbadNoiseCorrection=False, frameSize = frameSize)
        for frame in FrameGenerator(
            clip,
            frameSize=frameSize,
            hopSize=frameSize // 2,
        ):
            snr_instant, snr_ema_noisy, snr_spetral = snr(frame)
        snr_live.append(snr_ema_noisy)

    threshold = float (parameters.get("pValue", 0.05))
    pValue = stats.ks_2samp(snr_train, snr_live)[1]
    return int(pValue<threshold), pValue