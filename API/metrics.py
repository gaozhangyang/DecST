import numpy as np
from skimage.metrics import structural_similarity as cal_ssim

def MAE(pred, true):
    return np.mean(np.abs(pred-true),axis=(0,1)).sum()

def MSE(pred, true):
    return np.mean((pred-true)**2,axis=(0,1)).sum()

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true),axis=(0,1,2)).sum()

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true),axis=(0,1,2)).sum()

# def batch_psnr(gen_frames, gt_frames):
#     axis = (1, 2)
#     x = np.int32(gen_frames)
#     y = np.int32(gt_frames)
#     num_pixels = float(np.size(gen_frames[0]))
#     mse = np.sum((x - y)**2, axis=axis, dtype=np.float32) / num_pixels
#     psnr = 20 * np.log10(255) - 10 * np.log10(mse)
#     return np.mean(psnr)

# cite the `PSNR` code from E3d-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py
def PSNR(pred, true):
    # pred = np.maximum(pred, 0)
    # pred = np.minimum(pred, 1)
    mse = np.mean((np.uint8(pred * 255)-np.uint8(true * 255))**2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def metric(pred, true, mean,std, return_ssim_psnr=False, clip_range=[0, 1]):
    pred = pred*std + mean
    true = true*std + mean
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    rmse = -1
    mape = -1
    mspe = -1

    if return_ssim_psnr:
        # traffic 
        pred = np.maximum(pred, clip_range[0])
        pred = np.minimum(pred, clip_range[1])

        ssim = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2), true[b, f].swapaxes(0, 2), multichannel=True)
        ssim = ssim / (pred.shape[0] * pred.shape[1])

        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                # psnr += compare_psnr(np.uint8(pred[b, f] * 255), np.uint8(true[b, f] * 255))
                psnr += PSNR(pred[b, f], true[b, f])
                # psnr += batch_psnr(np.uint8(pred[b, f] * 255), np.uint8(true[b, f] * 255))
        psnr = psnr / (pred.shape[0] * pred.shape[1])

        return mae,mse,rmse,mape,mspe,ssim,psnr
    else:
        return mae,mse,rmse,mape,mspe 