function out = sbx_motion_correct(sbxFiles, varargin)
% SBX_MOTION_CORRECT  Rigid (+ optional non-rigid) motion correction of
% Scanbox .sbx recordings.
%
%   out = sbx_motion_correct(sbxFiles)
%   out = sbx_motion_correct(sbxFiles, 'Name', Value, ...)
%
% Reads .sbx files directly (no tif conversion), registers every frame to a
% template with whitened FFT phase correlation, and writes a uint16 BigTIFF.
% If several files are given (e.g. the segments 001_000, 001_001, ... of one
% recording) they are registered to a single template and written as one
% concatenated movie, in the order given.
%
% Algorithm (template init and non-rigid step follow suite2p's registration):
%   1. Template: sample nimgInit frames evenly across the recording, pick
%      the frame whose 20 best frame-frame correlations are highest, average
%      its 20 best-correlated frames, then refine for nIterInit iterations
%      (register the sample, average an increasing fraction of the best
%      aligned frames, re-centre).
%   2. Pass 1 (rigid estimate): for each batch of frames, taper edges, FFT,
%      whiten, multiply by the precomputed whitened + Gaussian-smoothed
%      template spectrum, inverse FFT, take the peak within +/-maxShift,
%      then refine to subpixel with a 3-point parabolic fit.
%   3. Clean-up: replace rigid shifts with weak correlation peaks or jumps
%      away from a moving median; choose the output crop from robust
%      percentiles.
%   4. Pass 2 (apply): if 'nonrigid' is on, rigidly shift the frames, split
%      them into overlapping blocks and repeat step 2 per block against the
%      matching template block (small search window). Blocks with a weak
%      peak (low peak-to-sidelobe SNR) borrow the correlation maps of their
%      neighbours. Block shifts are interpolated bilinearly between block
%      centres into a per-pixel shift field. Each raw frame is then warped
%      once by rigid + non-rigid shift (or just shifted, if rigid only),
%      cropped, and written as uint16.
%   Shift convention: corrected(y,x) = raw(y + dy + dyNR(y,x), x + dx + dxNR(y,x)),
%   with dyNR = dxNR = 0 for rigid-only correction.
%
% Speed: on a GPU (NVIDIA) batches run as gpuArrays (use canUseGPU()
% to check). Without one, and with
% the Parallel Computing Toolbox, a parallel pool (processes or threads;
% see 'poolProfile') is used: pass 1 is a
% parfor over batches (each worker reads its own frames from the .sbx and
% returns only shifts); pass 2 is a parfeval pipeline (workers read and
% shift the next batches while this MATLAB writes the tif in order).
% Without either, everything runs serially on the CPU (MATLAB still
% multithreads the FFTs).
%
% INPUT
%   sbxFiles  - path to one .sbx file (with or without extension), or a cell
%               array of paths to register together and concatenate.
%
% OPTIONAL NAME-VALUE INPUTS (defaults in brackets)
%   'outfile'        ([first file]_MC.tif) output tif path. Shifts and
%                    images are saved next to it as *_shifts.mat.
%   'channel'        ([] = auto) PMT channel to write (1 = green, 2 = red),
%                    same numbering as the _ch1/_ch2 tifs from sbx2tif.
%                    Auto reads the recorded channels from the Scanbox
%                    info (info.chan.save, e.g. [1 0 0 0] = green only,
%                    [0 1 0 0] = red only; or info.channels on old files)
%                    and picks the functional green channel (1) if every
%                    file has it, otherwise the one channel they share.
%   'alignChannel'   ([] = 'channel') PMT channel used to estimate motion.
%                    Use 2 if a structural red channel was recorded
%                    simultaneously.
%   'crop'           ([1 80 Inf Inf]) [row1 col1 row2 col2] crop applied to
%                    raw frames before anything else. Default drops the first
%                    79 columns, matching sbx2tif_lily2025 in the pipeline.
%   'dropFirstFrame' (true) drop the first frame of every file (it is often
%                    partial / bright), matching batch_2p_processData_Lily.
%   'maxFrames'      (Inf) only process the first N frames (for testing).
%   'maxShift'       (0.1) max rigid shift as a fraction of max(frame size)
%                    (suite2p maxregshift).
%   'smoothSigma'    (1.15) Gaussian smoothing (px) of the phase correlation
%                    (suite2p smooth_sigma).
%   'nimgInit'       (300) frames sampled to build the template.
%   'nIterInit'      (8) template refinement iterations.
%   'batchSize'      (500) frames per batch; reduced automatically on GPU
%                    to fit memory.
%   'subpixel'       (true) parabolic subpixel shifts + bilinear apply.
%                    false = integer shifts only (no interpolation blur;
%                    with 'nonrigid' the warp uses nearest-neighbour).
%   'rejectOutliers' (true) replace outlier rigid shifts by the moving median.
%   'outlierWindow'  (15) moving-median window (frames).
%   'outlierNMAD'    (6) jump threshold in robust SDs of the residual.
%   'outlierMinJump' (3) jumps below this many px are never outliers.
%   'minPeakRatio'   (0.3) frames whose correlation peak is below this
%                    fraction of the median peak are outliers.
%   'cropPrctile'    (0.5) output crop covers rigid shifts between this
%                    percentile and 100 minus it; more extreme frames get
%                    edge pixels replicated instead of shrinking the FOV.
%   'outCrop'        ([]) [row1 col1 row2 col2] output rectangle in RAW sbx
%                    frame pixels (the same convention as crop.out in the
%                    output). [] = the region valid for this run's own shifts,
%                    which differs between runs (a short reference moves less
%                    than 30000 frames of a session). Give the same rectangle
%                    to several runs, e.g. the green movie and the red
%                    reference, and their tifs cover exactly the same pixels.
%                    Frames that moved beyond it get edge pixels replicated.
%   'nonrigid'       (false) add piecewise-rigid (block) correction after
%                    the rigid step.
%   'blockSize'      ([128 128]) block size in px [rows cols] (or scalar).
%                    Blocks overlap; about 1.5*size/blockSize per dimension
%                    (suite2p make_blocks).
%   'maxShiftNR'     (5) max residual block shift in px after rigid.
%                    The output is cropped by this + 1 px on every side.
%   'snrThresh'      (1.2) blocks whose peak-to-sidelobe ratio is below this
%                    are smoothed with their neighbours (suite2p snr_thresh).
%   'nrNeighborSigma'(1) Gaussian width, in block units, of that neighbour
%                    smoothing.
%   'useGPU'         ([] = auto) use gpuArray if available.
%   'parallel'       ([] = auto: on when not using the GPU and the Parallel
%                    Computing Toolbox is available) use a parallel pool on
%                    the CPU; uses the open pool or starts one. Batch size
%                    is reduced so all workers fit in RAM.
%   'nWorkers'       ([] = pool default, usually the number of physical
%                    cores) workers to start if no pool is open.
%   'poolProfile'    ([] = default profile) profile to start the pool with
%                    if none is open, e.g. 'Threads' or 'Processes'. An open
%                    pool of any kind is reused. Each pass is first tried on
%                    a 2-frame batch; if the pool cannot run it (e.g. a
%                    function unsupported on thread workers) that pass runs
%                    serially instead.
%   'writeOutput'    (true) false = only estimate and save shifts.
%   'verbose'        (true) print progress.
%   'bidiShift'      (0) correction of the bidirectional scan phase error
%                    (even and odd lines displaced in x relative to each
%                    other, a comb on vertical edges that blurs the mean
%                    image and that motion correction cannot remove).
%                      0       off (the offset is still estimated and
%                              recorded in out.bidi, but not applied)
%                      'auto'  estimate from the data and apply
%                      number  apply this lag (px) = out.bidi.lag
%                              convention below
%                    Applied when the frames are read, before the template,
%                    motion estimation and the output tif, on the
%                    full-width rows (before 'crop'). Each scan direction
%                    is moved by half the lag in opposite directions
%                    (Lanczos-3 sub-pixel interpolation along the row), so
%                    the image stays centred. The same shift is used for
%                    every channel read.
%   'bidiNFrames'    (200) frames, evenly spaced over the recording, used
%                    to estimate the lag.
%   'bidiMaxLag'     (8) largest |lag| (px) searched.
%   'bidiBands'      (4) number of column bands for the per-band lag
%                    (a lag that changes across x leaves a residual comb).
%   'diagnostics'    (false) true = draw the diagnostic figures (see
%                    mc_diag_method.session) from the frames this function already
%                    has in memory during pass 2, with no re-reading of the
%                    .sbx or the tif. Collected during pass 2: mean F and the
%                    correlation with the template of every frame (raw and
%                    corrected), mean images of time bins, and a sample of
%                    frames (raw and corrected) for the PC1 maps. Needs
%                    writeOutput. 'raw' here is after the scan-phase
%                    correction, before motion correction. Memory: two copies
%                    of the sample (2 x nSample frames of the output size).
%   'diagDir'        ('' = <outfile folder>\mc_diagnostics) figure folder.
%   'diagOpts'       ({}) name-value options of mc_diag_method.session: 'name',
%                    'nSample' (here 1000 if not given), 'nBins', 'nPC',
%                    'segNPC', 'zoomSize', 'smoothWin'.
%
% OUTPUT (struct, also saved to *_shifts.mat)
%   dy, dx      - [nFrames x 1] rigid shifts applied (px, after clean-up)
%   dyRaw, dxRaw- [nFrames x 1] rigid shifts before clean-up
%   cmax        - [nFrames x 1] phase-correlation peak (alignment quality)
%   bad         - [nFrames x 1] logical, frames flagged as outliers
%   nonrigid    - [] if 'nonrigid' is off, otherwise a struct with
%                   dy, dx     [nFrames x nBlocks] single, residual block
%                              shifts after the rigid shift
%                   lowSNR     [nFrames x nBlocks] logical, blocks that were
%                              smoothed with their neighbours
%                   gridSize   [nBlocksY nBlocksX]; block j is at grid
%                              position (mod(j-1,nY)+1, floor((j-1)/nY)+1)
%                   centersY, centersX  block centres (cropped-frame px)
%                   blockRows, blockCols [nBlocks x blockSize] pixel indices
%                              of each block (cropped-frame px)
%                   blockSize, maxShift
%   refImg      - template, covering rows params.estRows / cols
%                 params.estCols of the cropped frame
%   meanImg, maxImg - mean / max projection of the registered movie
%   outRows, outCols - rows/cols of the cropped frame kept in the output
%   crop        - where the output tif sits in the raw sbx frame:
%                   rawSize     [rows cols] of the raw sbx frame
%                   raw         [row1 col1 row2 col2] 'crop' applied to the
%                               raw frame (scan-turnaround columns etc.)
%                   out         [row1 col1 row2 col2] of the output tif in
%                               raw-frame px (raw crop + valid-region crop)
%                   outSize     [rows cols] of the output tif
%   frameIndex  - [nFrames x 2] [file #, 0-based frame # in that .sbx]
%   bidi        - bidirectional scan phase offset, estimated on the raw
%                 frames (always recorded, applied only if 'bidiShift'):
%                   mode        'off' | 'auto' | 'manual'
%                   lag         px, global estimate BEFORE correction.
%                               CONVENTION: lag > 0 = features of the odd
%                               rows (1,3,5.. of the raw frame) lie lag
%                               columns to the RIGHT of the same features
%                               in the even rows (2,4,6..).
%                   applied     [shiftOdd shiftEven] px actually applied;
%                               out(:,c) = in(:,c - shift) (+ = to the
%                               right). [0 0] = nothing applied.
%                               Applied shifts = [-lag/2, +lag/2].
%                   lagMedian, lagMAD  per-frame lags: median and MAD
%                   lagBands    lag per column band (before correction)
%                   bandCols    [nBands x 2] first/last raw-frame column of
%                               each band
%                   lags, cc    lags searched (px) and mean correlation
%                               (all columns) before correction
%                   lagAfter, lagBandsAfter, ccAfter   the same measured on
%                               the corrected frames (residual; ~0 = good)
%                   channel, nFrames, rows (raw-frame row range used),
%                   reliable    false if the estimate was not trusted
%                               (low correlation, or peak at the search
%                               limit); an 'auto' estimate that is not
%                               reliable is not applied.
%   files, outfile, params, timeElapsed
%               params.bidiShiftUsed = lag actually applied (px, 0 = off).
%
% EXAMPLE
%   f = dir(fullfile(dateDir, '250314_001_*.sbx'));
%   out = sbx_motion_correct(fullfile(dateDir, {f.name}), ...
%       'outfile', fullfile(dateDir, 'lintrack_rec001', 'ConcatenatedMovie_MC.tif'), ...
%       'nonrigid', true);

timerStart = tic;

%%%%%%%%%%%%%%
%%% INPUTS %%%
%%%%%%%%%%%%%%
sbxFiles = regexprep(cellstr(sbxFiles), '\.sbx$', '', 'ignorecase');
sbxFiles = sbxFiles(:)';

defaults = {            % name, default (see help above)
    'outfile',          [sbxFiles{1} '_MC.tif']
    'channel',          []
    'alignChannel',     []
    'crop',             [1 80 Inf Inf]
    'dropFirstFrame',   true
    'maxFrames',        Inf
    'maxShift',         0.1
    'smoothSigma',      1.15
    'nimgInit',         300
    'nIterInit',        8
    'batchSize',        500
    'subpixel',         true
    'rejectOutliers',   true
    'outlierWindow',    15
    'outlierNMAD',      6
    'outlierMinJump',   3
    'minPeakRatio',     0.3
    'cropPrctile',      0.5
    'outCrop',          []
    'nonrigid',         true % default non-rigid mc
    'blockSize',        [128 128]
    'maxShiftNR',       5
    'snrThresh',        1.2
    'nrNeighborSigma',  1
    'useGPU',           []
    'parallel',         []
    'nWorkers',         []
    'poolProfile',      []
    'writeOutput',      true
    'verbose',          true
    'bidiShift',        0
    'bidiNFrames',      200
    'bidiMaxLag',       8
    'bidiBands',        4
    'diagnostics',      false
    'diagDir',          ''
    'diagOpts',         {}
    };
ip = inputParser;
ip.FunctionName = mfilename;
for i = 1:size(defaults, 1)
    ip.addParameter(defaults{i, :});
end
ip.parse(varargin{:});
p = ip.Results;
p.outfile = char(p.outfile);
if isempty(p.useGPU)
    try
        p.useGPU = canUseGPU();
    catch
        p.useGPU = false;
    end
end
if isempty(p.parallel)
    p.parallel = ~p.useGPU && license('test', 'Distrib_Computing_Toolbox') && exist('parpool', 'file') > 0;
end
pool = [];
if p.parallel && ~p.useGPU
    pool = startPool(p.nWorkers, p.poolProfile);
end
nW = 0;                                  % 0 = serial (parfor runs on this MATLAB)
if ~isempty(pool)
    nW = pool.NumWorkers;
end
p.parallel = nW > 0;
p.nWorkers = nW;
vprint = @(varargin) p.verbose && fprintf(varargin{:});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FILE INFO & INDEXING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
src = cellfun(@sbx_method.info, sbxFiles, 'UniformOutput', false);
src = [src{:}];
if isempty(p.channel)
    % auto: functional green (1) if recorded in every file, else the one
    % channel all files share (e.g. a red-only reference)
    common = src(1).chs;
    for f = 2:numel(src)
        common = intersect(common, src(f).chs);
    end
    if ismember(1, common)
        p.channel = 1;
    elseif isscalar(common)
        p.channel = common;
        warning('%s: no green channel recorded; using channel %d.', src(1).path, common);
    else
        error('No channel recorded in all files (channels: %s); set ''channel''.', ...
            strjoin(arrayfun(@(s) mat2str(s.chs), src, 'UniformOutput', false), ', '));
    end
    vprint('Channel (auto): %d  [recorded: %s]\n', p.channel, mat2str(src(1).chs));
end
if isempty(p.alignChannel)
    p.alignChannel = p.channel;
end
for f = find(arrayfun(@(s) ~all(ismember([p.channel p.alignChannel], s.chs)), src))
    error('%s: channel %d/%d not recorded (channels: %s)', src(f).path, ...
        p.channel, p.alignChannel, mat2str(src(f).chs));
end
if any([src.rows] ~= src(1).rows | [src.cols] ~= src(1).cols)
    error('Frame size differs between the .sbx files.');
end
first = double(logical(p.dropFirstFrame));             % 0-based first frame used
nUse = max([src.nFramesDisk] - first, 0);
nUse = min(nUse, max(p.maxFrames - [0 cumsum(nUse(1:end-1))], 0));   % cap total (testing)
[src.first] = deal(first);
nUseC = num2cell(nUse);
[src.n] = nUseC{:};
cum = [0 cumsum(nUse)];
N = cum(end);
if N < 2
    error('Fewer than 2 frames to register.');
end
fileOfFrame = repelem((1:numel(src))', nUse(:));   % row if one file, column if several
frameIndex = [fileOfFrame(:), ...
    cell2mat(arrayfun(@(n) (0:n-1)', nUse(:), 'UniformOutput', false)) + first];

% raw crop, clipped to frame size
crop = p.crop(:)';
rc = [max(crop(1:2), 1), min(crop(3:4), [src(1).rows src(1).cols])];
H = rc(3) - rc(1) + 1;
W = rc(4) - rc(2) + 1;
io = struct('src', src, 'cum', cum, 'rc', rc, 'useGPU', p.useGPU, 'bidi', [0 0]);

% bidirectional scan phase: always estimated on the raw frames and recorded;
% applied (io.bidi, used by every frame read below) if 'bidiShift' asks for it
[bidi, io.bidi] = resolveBidi(io, p, N, vprint);
p.bidiShiftUsed = io.bidi(2) - io.bidi(1);       % = lag applied (0 = off)

% estimation region: central crop to FFT-friendly sizes (factors 2,3,5,7)
He = fastFFTSize(H);
We = fastFFTSize(W);
er = floor((H - He)/2) + (1:He);
ec = floor((W - We)/2) + (1:We);
lcorr = max(1, min(round(p.maxShift * max(H, W)), floor(min(He, We)/2) - 1));

% non-rigid block layout (on the estimation region)
NR = [];
if p.nonrigid
    NR = blockGeometry(He, We, p);
end

% batch sizes: keep ~6-8 single-size copies of a batch in memory (per
% worker when running in parallel)
if p.useGPU
    g = gpuDevice();
    memBudget = 0.5 * g.AvailableMemory;
    B = max(16, min(p.batchSize, floor(memBudget / (He * We * 4 * 6))));
else
    memBudget = 4e9;
    try
        m = memory;                      % Windows only
        memBudget = 0.5 * m.MemAvailableAllArrays;
    catch
    end
    memBudget = memBudget / max(nW, 1);
    B = max(16, min(p.batchSize, floor(memBudget / (H * W * 4 * 8))));
end
nBatch = ceil(N / B);
batches = arrayfun(@(b) (b-1)*B+1 : min(b*B, N), 1:nBatch, 'UniformOutput', false);
if p.nonrigid
    NR.subBatch = max(1, min(B, floor(memBudget / (prod(NR.bs) * NR.nB * 4 * 6))));
end

vprint('sbx_motion_correct: %d file(s), %d frames, %dx%d px (est. %dx%d), maxShift %d px, GPU %d, workers %d, batch %d\n', ...
    numel(src), N, H, W, He, We, lcorr, p.useGPU, nW, B);
if p.nonrigid
    vprint('  non-rigid: %dx%d blocks of %dx%d px, maxShiftNR %d px\n', ...
        NR.nby, NR.nbx, NR.bs(1), NR.bs(2), NR.lcorr);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1. TEMPLATE (suite2p init) %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tStep = tic;
sampleIdx = unique(round(linspace(1, N, min(p.nimgInit, N))));
F = loadFrames(io, sampleIdx, p.alignChannel);
refImg = initReference(F(er, ec, :), p, lcorr);
clear F
R = makeRef(refImg, p.smoothSigma);
if p.nonrigid
    NR = blockReferences(NR, refImg, p.smoothSigma);
end
vprint('  template from %d frames: %.1f s\n', numel(sampleIdx), toc(tStep));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2. PASS 1: ESTIMATE RIGID SHIFTS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Each worker reads its own batch and returns only the shifts. With nW = 0
% the parfor runs serially here (no toolbox needed; GPU path included).
tStep = tic;
alignCh = p.alignChannel;
subpix = p.subpixel;
nW1 = poolWorks(pool, 'pass 1', @probePass1, io, batches{1}(1:min(2, end)), alignCh, er, ec, R, lcorr, subpix);
q = progressQueue(p.verbose, nW1, 'estimate', nBatch);
[dyC, dxC, cC] = deal(cell(1, nBatch));
parfor (bb = 1:nBatch, nW1)
    Y = loadFrames(io, batches{bb}, alignCh);
    [dyC{bb}, dxC{bb}, cC{bb}] = phaseCorr(Y(er, ec, :), R, lcorr, subpix);
    reportProgress(q, bb);
end
dyRaw = vertcat(dyC{:});
dxRaw = vertcat(dxC{:});
cmax = vertcat(cC{:});
clear dyC dxC cC
vprint('  rigid shifts estimated: %.1f s\n', toc(tStep));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 3. OUTLIER CLEAN-UP & OUTPUT CROP %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dy = dyRaw;
dx = dxRaw;
bad = false(N, 1);
if p.rejectOutliers
    dyMed = movmedian(dyRaw, p.outlierWindow);
    dxMed = movmedian(dxRaw, p.outlierWindow);
    res = hypot(dyRaw - dyMed, dxRaw - dxMed);
    bad = res > max(p.outlierNMAD * 1.4826 * median(res), p.outlierMinJump) | ...
          cmax < p.minPeakRatio * median(cmax);
    dy(bad) = dyMed(bad);
    dx(bad) = dxMed(bad);
    if ~p.subpixel
        dy = round(dy);
        dx = round(dx);
    end
    vprint('  %d/%d frames flagged as outliers (%.2f%%)\n', sum(bad), N, 100*mean(bad));
end
% block shifts add up to NR.lcorr (+0.5 subpixel) px on top of the rigid
% shift, and bilinear sampling reads one px further
margin = 0;
if p.nonrigid
    margin = NR.lcorr + 1;
end
outRows = validRange(dy, H, p.cropPrctile, margin);
outCols = validRange(dx, W, p.cropPrctile, margin);
if ~isempty(p.outCrop)
    % forced output rectangle (raw-frame pixels): the same pixels in every run
    oc = double(p.outCrop(:)');
    if numel(oc) ~= 4 || any(~isfinite(oc)) || oc(3) < oc(1) || oc(4) < oc(2)
        error('outCrop must be [row1 col1 row2 col2] in raw-frame pixels.');
    end
    forcedRows = (oc(1):oc(3)) - (rc(1) - 1);        % into the cropped frame
    forcedCols = (oc(2):oc(4)) - (rc(2) - 1);
    if forcedRows(1) < 1 || forcedCols(1) < 1 || forcedRows(end) > H || forcedCols(end) > W
        error(['outCrop [%d %d %d %d] is outside the cropped frame ' ...
            '(raw rows %d-%d, cols %d-%d after ''crop'').'], oc, rc(1), rc(3), rc(2), rc(4));
    end
    if isempty(outRows) || isempty(outCols) || forcedRows(1) < outRows(1) || ...
            forcedRows(end) > outRows(end) || forcedCols(1) < outCols(1) || forcedCols(end) > outCols(end)
        vprint(['  note: outCrop reaches beyond the region valid for this movie''s shifts; ' ...
            'edge pixels are replicated in the frames that moved most\n']);
    end
    outRows = forcedRows;
    outCols = forcedCols;
end
if isempty(outRows) || isempty(outCols)
    error('Shifts are larger than the frame: no valid output region.');
end
if p.nonrigid
    % block centres in cropped-frame coordinates, and the matrices that
    % interpolate block shifts onto the output pixel grid
    NR.ycFull = er(1) - 1 + NR.yc;
    NR.xcFull = ec(1) - 1 + NR.xc;
    NR.Ay = toDevice(linInterpMatrix(outRows, NR.ycFull), p.useGPU);
    NR.Ax = toDevice(linInterpMatrix(outCols, NR.xcFull), p.useGPU);
    [Xo, Yo] = meshgrid(outCols, outRows);
    NR.Xo = toDevice(Xo, p.useGPU);
    NR.Yo = toDevice(Yo, p.useGPU);
    clear Xo Yo
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 4. PASS 2: NON-RIGID ESTIMATE (optional), APPLY SHIFTS, WRITE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meanImg = [];
maxImg = [];
nonrigidOut = [];
dgn = [];                            % diagnostics collectors (p.diagnostics)
if p.nonrigid
    [dyNR, dxNR] = deal(zeros(N, NR.nB, 'single'));
    lowSNR = false(N, NR.nB);
end
warpMethod = 'nearest';
if p.subpixel
    warpMethod = 'linear';
end
if p.writeOutput || p.nonrigid
    tStep = tic;
    if p.writeOutput
        tw = openTiff(p.outfile);
        tags = tiffTags(numel(outRows), numel(outCols));
        closeTiff = onCleanup(@() tw.close());
        sumImg = zeros(numel(outRows), numel(outCols));
        maxImg = zeros(numel(outRows), numel(outCols), 'uint16');
    end
    % what each batch needs (see pass2Batch); in parallel the block data is
    % sent to every worker once instead of with every batch
    s2 = struct('alignChannel', p.alignChannel, 'channel', p.channel, 'nonrigid', p.nonrigid, ...
        'writeOutput', p.writeOutput, 'subpixel', p.subpixel, 'er', er, 'ec', ec, ...
        'He', He, 'We', We, 'outRows', outRows, 'outCols', outCols, 'warpMethod', warpMethod);
    s2.NR = NR;
    % statistics for the diagnostic figures, collected from the frames that
    % are in memory anyway (see batchDiag)
    s2.diag = [];
    dgn = [];
    if p.diagnostics
        if p.writeOutput
            [dgn, s2.diag] = initDiag(p, N, outRows, outCols, er, ec, gather(refImg));
        else
            warning('sbx_motion_correct:diag', '''diagnostics'' needs writeOutput; skipped.');
        end
    end
    if nW > 0 && p.nonrigid && ~isa(pool, 'parallel.ThreadPool')
        s2.NR = parallel.pool.Constant(NR);   % thread workers share memory: not needed
    end
    g1 = batches{1}(1:min(2, end));
    nW2 = poolWorks(pool, 'pass 2', @pass2Batch, io, g1, dy(g1), dx(g1), s2);
    if nW2 == 0
        s2.NR = NR;
    end
    % serial: process batch b here. parallel: keep nW2+1 batches in flight on
    % the workers and collect them in order, so writing overlaps with reading
    % and shifting the next batches.
    futs = cell(1, nBatch);
    nSubmitted = 0;
    try
        for b = 1:nBatch
            gi = batches{b};
            if nW2 > 0
                while nSubmitted < min(nBatch, b + nW2)
                    nSubmitted = nSubmitted + 1;
                    gs = batches{nSubmitted};
                    futs{nSubmitted} = parfeval(pool, @pass2Batch, 5, io, gs, dy(gs), dx(gs), s2);
                end
                [Z, bdy, bdx, blow, dg] = fetchOutputs(futs{b});
                futs{b} = [];
            else
                [Z, bdy, bdx, blow, dg] = pass2Batch(io, gi, dy(gi), dx(gi), s2);
            end
            if ~isempty(dgn)
                dgn = collectDiag(dgn, gi, dg);
            end
            if p.nonrigid
                dyNR(gi, :) = bdy';
                dxNR(gi, :) = bdx';
                lowSNR(gi, :) = blow';
            end
            if p.writeOutput
                sumImg = sumImg + sum(Z, 3, 'double');
                maxImg = max(maxImg, max(Z, [], 3));
                for k = 1:size(Z, 3)
                    if gi(k) > 1
                        tw.writeDirectory();
                    end
                    tw.setTag(tags);
                    tw.write(Z(:, :, k));
                end
            end
            vprint('  pass 2: batch %d/%d\n', b, nBatch);
        end
    catch ME
        cellfun(@cancel, futs(~cellfun(@isempty, futs)));   % stop queued work
        rethrow(ME);
    end
    if p.writeOutput
        clear closeTiff
        meanImg = sumImg / N;
    end
    vprint('  pass 2 done: %.1f s\n', toc(tStep));
    if p.nonrigid
        vprint('  non-rigid: %.1f%% of block estimates smoothed (low SNR)\n', 100*mean(lowSNR(:)));
        nonrigidOut = struct('dy', dyNR, 'dx', dxNR, 'lowSNR', lowSNR, ...
            'gridSize', [NR.nby NR.nbx], 'centersY', NR.ycFull, 'centersX', NR.xcFull, ...
            'blockRows', NR.rows + er(1) - 1, 'blockCols', NR.cols + ec(1) - 1, ...
            'blockSize', NR.bs, 'maxShift', NR.lcorr);
    end
end

%%%%%%%%%%%%%%
%%% OUTPUT %%%
%%%%%%%%%%%%%%
p.crop = rc;
p.estRows = er;
p.estCols = ec;
p.maxShiftPx = lcorr;
p.batchSize = B;
out = struct('dy', dy, 'dx', dx, 'dyRaw', dyRaw, 'dxRaw', dxRaw, 'cmax', cmax, 'bad', bad, ...
    'nonrigid', nonrigidOut, 'refImg', gather(refImg), 'meanImg', meanImg, 'maxImg', maxImg, ...
    'outRows', outRows, 'outCols', outCols, 'crop', cropInfo(src(1), rc, outRows, outCols), ...
    'frameIndex', frameIndex, 'bidi', bidi, 'files', {strcat(sbxFiles, '.sbx')}, 'outfile', p.outfile, 'params', p, ...
    'timeElapsed', toc(timerStart));
[od, on] = fileparts(p.outfile);
save(fullfile(od, [on '_shifts.mat']), '-struct', 'out', '-v7.3');
vprint('sbx_motion_correct done in %.1f s -> %s\n', out.timeElapsed, p.outfile);

% diagnostic figures from the data kept during pass 2 (nothing is re-read)
if ~isempty(dgn)
    try
        diagDir = p.diagDir;
        if isempty(diagDir)
            diagDir = fullfile(od, 'mc_diagnostics');
        end
        cnt = reshape(max(dgn.cnt, 1), 1, 1, []);
        data = struct('corSample', dgn.Xcor, 'rawSample', dgn.Xraw, 'sampleIdx', dgn.sampleIdx, ...
            'meanF', struct('cor', dgn.meanFcor, 'raw', dgn.meanFraw), ...
            'frameCorr', struct('cor', dgn.cCor, 'raw', dgn.cRaw), 'corrRef', 'template', ...
            'binImg', struct('cor', single(dgn.sumCor ./ cnt), 'raw', single(dgn.sumRaw ./ cnt)), ...
            'segNames', {regexprep(sbxFiles, '.*[\\/]', '')}, ...
            'rawNote', 'raw = scan-phase corrected, no motion correction', ...
            'shifts', struct('dy', dy, 'dx', dx, 'cmax', cmax, 'bad', bad, 'nonrigid', {nonrigidOut}, ...
                'frameIndex', frameIndex, 'crop', out.crop, 'bidi', bidi, 'outRows', outRows, ...
                'outCols', outCols, 'params', p));
        name = optGet(p.diagOpts, 'name', on);
        opts = p.diagOpts;
        k = find(strcmpi(opts(1:2:end), 'name'), 1);
        if ~isempty(k)
            opts(2*k-1:2*k) = [];
        end
        mc_diag_method.session(data, [], diagDir, 'name', name, opts{:});
    catch ME
        warning('sbx_motion_correct:diag', 'Diagnostic figures failed: %s', ME.message);
    end
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% BIDIRECTIONAL SCAN PHASE (even/odd) %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [bidi, shifts] = resolveBidi(io, p, N, vprint)
% Estimate the even/odd-row offset on the raw frames, decide what to apply
% from p.bidiShift, verify the residual after correction, and return the
% documentation struct (saved as out.bidi) and the [odd even] row shifts.
mode = 'off';
if ischar(p.bidiShift) || isstring(p.bidiShift)
    if ~strcmpi(char(p.bidiShift), 'auto')
        error('''bidiShift'' must be 0, a number (px) or ''auto''.');
    end
    mode = 'auto';
elseif isnumeric(p.bidiShift) && isscalar(p.bidiShift) && isfinite(p.bidiShift)
    if p.bidiShift ~= 0
        mode = 'manual';
    end
else
    error('''bidiShift'' must be 0, a number (px) or ''auto''.');
end

bidi = sbx_method.bidiOffset({io.src.path}, 'channel', p.alignChannel, 'crop', io.rc, ...
    'first', io.src(1).first, 'nFrames', p.bidiNFrames, 'maxLag', p.bidiMaxLag, ...
    'bands', p.bidiBands, 'shifts', [0 0]);
bidi.mode = mode;
lag = 0;
switch mode
    case 'auto'
        if bidi.reliable
            lag = bidi.lag;
        else
            warning('sbx_motion_correct:bidi', ...
                'bidiShift auto: estimate not reliable (lag %.2f px, peak corr %.2f); not applied.', ...
                bidi.lag, max(bidi.cc(:)));
        end
    case 'manual'
        lag = double(p.bidiShift);
end
shifts = [-lag/2, lag/2];
bidi.applied = shifts;
bidi.lagAfter = NaN;
bidi.lagBandsAfter = nan(size(bidi.lagBands));
bidi.ccAfter = [];
if lag ~= 0
    A = sbx_method.bidiOffset({io.src.path}, 'channel', p.alignChannel, 'crop', io.rc, ...
    'first', io.src(1).first, 'nFrames', p.bidiNFrames, 'maxLag', p.bidiMaxLag, ...
    'bands', p.bidiBands, 'shifts', shifts);
    bidi.lagAfter = A.lag;
    bidi.lagBandsAfter = A.lagBands;
    bidi.ccAfter = A.cc;
end
bidi.convention = ['lag > 0: features of odd raw rows lie lag columns right of the same features ' ...
    'in even rows. applied = [shiftOdd shiftEven] px, out(:,c) = in(:,c-shift) (+ = right) = [-lag/2 +lag/2].'];

vprint('Bidirectional scan offset (even vs odd rows): %+.2f px (per-frame median %+.2f, MAD %.2f; bands %s)%s\n', ...
    bidi.lag, bidi.lagMedian, bidi.lagMAD, mat2str(round(bidi.lagBands, 2)), ...
    ternary(bidi.reliable, '', '  [NOT RELIABLE]'));
if lag ~= 0
    vprint('  applied %+.2f px (odd rows %+.2f, even rows %+.2f); residual after correction %+.2f px (bands %s)\n', ...
        lag, shifts(1), shifts(2), bidi.lagAfter, mat2str(round(bidi.lagBandsAfter, 2)));
else
    vprint('  not applied (bidiShift = %s)\n', ternary(strcmp(mode, 'off'), '0', 'auto, unreliable'));
end
end

function s = ternary(c, a, b)
if c
    s = a;
else
    s = b;
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% RIGID REGISTRATION HELPERS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function refImg = initReference(F, p, lcorr)
% suite2p pick_initial_reference + compute_reference.
% F: [H W K] single sample frames (modified in place as they are aligned).
K = size(F, 3);
nTop = min(20, K);

% seed: frame with the highest mean correlation to its best 20 neighbours,
% averaged with its 20 best-correlated frames
Fm = reshape(F, [], K);
Fm = Fm - mean(Fm, 1);
cc = Fm' * Fm;
nd = sqrt(diag(cc));
cc = cc ./ (nd * nd');
clear Fm
ccSorted = sort(cc, 2, 'descend');
[~, iMax] = max(mean(ccSorted(:, 2:nTop), 2));   % skip self-correlation
[~, order] = sort(cc(iMax, :), 'descend');
refImg = mean(F(:, :, order(1:nTop)), 3);

% refine: align the sample to the template, average an increasing fraction
% of the best-aligned frames, re-centre on their mean shift
for it = 1:p.nIterInit
    [dy, dx, cm] = phaseCorr(F, makeRef(refImg, p.smoothSigma), lcorr, false);
    for k = 1:K
        F(:, :, k) = circshift(F(:, :, k), -[dy(k) dx(k)]);
    end
    [~, iSort] = sort(cm, 'descend');
    iSort = iSort(2:max(2, floor(K * it / (2 * p.nIterInit))));   % suite2p drops the single best frame
    refImg = circshift(mean(F(:, :, iSort), 3), [round(mean(dy(iSort))) round(mean(dx(iSort)))]);
end
end

function R = makeRef(refImg, smoothSigma)
% Precompute everything that depends only on the template.
[H, W] = size(refImg);
R.maskMul = spatialTaper(3 * smoothSigma, H, W, refImg);
R.maskOffset = mean(refImg(:)) * (1 - R.maskMul);
cf = conj(fft2(refImg));
R.cfRef = cf ./ (1e-5 + abs(cf)) .* gaussianFFT(smoothSigma, H, W, refImg);
end

function cc = whitenedXcorr(Y, maskMul, maskOffset, cfRef)
% Phase correlation maps of every 2-D page of Y against a template
% spectrum: taper edges toward the mean, FFT (batched over pages), whiten,
% multiply by the whitened + smoothed template spectrum, inverse FFT.
X = fft2(Y .* maskMul + maskOffset);
cc = real(ifft2(X ./ (1e-5 + abs(X)) .* cfRef));
end

function idx = lagIdx(n, l)
% Indices of lags -l..l in an FFT-ordered axis of length n (zero lag at l+1).
idx = [n-l+1:n, 1:l+1];
end

function [dy, dx, cmax] = phaseCorr(Y, R, lcorr, subpixel)
% Whitened phase correlation of every page of Y against the template.
% Returns shifts such that Y(y,x) ~ ref(y - dy, x - dx).
[H, W, n] = size(Y);
cc = whitenedXcorr(Y, R.maskMul, R.maskOffset, R.cfRef);
cc = reshape(cc(lagIdx(H, lcorr), lagIdx(W, lcorr), :), (2*lcorr+1)^2, n);
[dy, dx, cmax] = peakSubpixel(cc, lcorr, subpixel);
end

function [dy, dx, cmax] = peakSubpixel(cc, lcorr, subpixel)
% cc: [L*L x m] correlation windows (column-major L x L, zero lag at the
% centre). Returns the peak lag of each column, refined by a 3-point
% parabolic fit along each axis if subpixel is true.
L = 2 * lcorr + 1;
m = size(cc, 2);
[cmax, k] = max(cc, [], 1);
iy = mod(k - 1, L) + 1;
ix = floor((k - 1) / L) + 1;
dy = iy - lcorr - 1;
dx = ix - lcorr - 1;
if subpixel
    iyc = min(max(iy, 2), L - 1);
    ixc = min(max(ix, 2), L - 1);
    i0 = iyc + (ixc - 1) * L + (0:m-1) * L * L;
    c0 = cc(i0);
    sy = parabolicOffset(cc(i0 - 1), c0, cc(i0 + 1));
    sx = parabolicOffset(cc(i0 - L), c0, cc(i0 + L));
    sy(iy ~= iyc) = 0;    % peak on the window edge: no refinement
    sx(ix ~= ixc) = 0;
    dy = dy + sy;
    dx = dx + sx;
end
dy = double(gather(dy(:)));
dx = double(gather(dx(:)));
cmax = double(gather(cmax(:)));
end

function s = parabolicOffset(cm, c0, cp)
% Vertex of the parabola through (-1,cm), (0,c0), (+1,cp).
den = cm - 2 * c0 + cp;
s = (cm - cp) ./ (2 * den);
s(~(den < 0)) = 0;         % not a local maximum (or NaN)
s = max(min(s, 0.5), -0.5);
end

function c = cropInfo(s, rc, outRows, outCols)
% Output-tif position in raw sbx pixels (see OUTPUT: crop).
c = struct('rawSize', [s.rows s.cols], 'raw', rc, ...
    'out', [rc(1)-1+outRows(1), rc(2)-1+outCols(1), rc(1)-1+outRows(end), rc(2)-1+outCols(end)], ...
    'outSize', [numel(outRows) numel(outCols)]);
end

function Z = applyShifts(Y, dy, dx, outRows, outCols)
% corrected(r,c) = Y(r+dy, c+dx), separable bilinear, computed only on the
% output crop. Out-of-range samples replicate the nearest edge pixel.
[H, W, n] = size(Y);
Z = zeros(numel(outRows), numel(outCols), n, 'like', Y);
for k = 1:n
    ny = floor(dy(k)); a = dy(k) - ny;
    nx = floor(dx(k)); b = dx(k) - nx;
    r0 = min(max(outRows + ny, 1), H);
    c0 = min(max(outCols + nx, 1), W);
    I = Y(:, :, k);
    Ir = (1 - a) * I(r0, :) + a * I(min(r0 + 1, H), :);
    Z(:, :, k) = (1 - b) * Ir(:, c0) + b * Ir(:, min(c0 + 1, W));
end
end

function r = validRange(d, len, pct, margin)
% Output rows (or cols) valid for all shifts between the pct and 100-pct
% percentiles: corrected(r) = raw(r + d) needs 1 <= r+floor(d), r+ceil(d) <= len.
% margin: extra px removed on both sides.
r = (1 + max(0, ceil(-prctileBase(d, pct))) + margin) : ...
    (len - max(0, ceil(prctileBase(d, 100 - pct))) - margin);
end

function mask = spatialTaper(sig, H, W, like)
% suite2p spatial_taper: sigmoid fall-off over the outer ~2*sig px.
sigmoid = @(u, n) 1 ./ (1 + exp((abs(u - (n-1)/2) - ((n-1)/2 - 2*sig)) / sig));
mask = cast(sigmoid((0:H-1)', H) .* sigmoid(0:W-1, W), 'like', like);
end

function S = gaussianFFT(sig, H, W, like)
% Fourier transform of a unit-sum Gaussian (std sig px), FFT ordering.
freq = @(n) [0:floor((n-1)/2), -floor(n/2):-1] / n;
S = cast(exp(-2 * pi^2 * sig^2 * (freq(H)'.^2 + freq(W).^2)), 'like', like);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% NON-RIGID (BLOCK) REGISTRATION %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function NR = blockGeometry(He, We, p)
% Overlapping blocks tiling the He x We estimation region (suite2p
% make_blocks). Block j sits at grid position (IY(j), IX(j)), column-major.
bs = round(p.blockSize(:)') .* [1 1];     % scalar -> [bs bs]
bs = min(bs, [He We]);
if min(bs) < 16
    error('blockSize must be at least 16 px.');
end
[ys, NR.yc] = blockStarts(He, bs(1));
[xs, NR.xc] = blockStarts(We, bs(2));
NR.nby = numel(ys);
NR.nbx = numel(xs);
NR.nB = NR.nby * NR.nbx;
NR.bs = bs;
[IY, IX] = ndgrid(1:NR.nby, 1:NR.nbx);
% reshape: ys(IY(:)) is a row vector when ys is a row, but takes the shape of
% the index when ys is a scalar (a single block row/column, e.g. strips)
NR.rows = reshape(ys(IY(:)), [], 1) + (0:bs(1)-1);       % [nB x bs(1)] pixel rows of each block
NR.cols = reshape(xs(IX(:)), [], 1) + (0:bs(2)-1);
% neighbour weights for smoothing low-SNR blocks (rows sum to 1)
Wn = exp(-((IY(:) - IY(:)').^2 + (IX(:) - IX(:)').^2) / (2 * p.nrNeighborSigma^2));
NR.Wn = Wn ./ sum(Wn, 2);
NR.lcorr = max(3, min(round(p.maxShiftNR), floor(min(bs)/2) - 1));
NR.snrThresh = p.snrThresh;
end

function [starts, centres] = blockStarts(L, b)
% ~1.5*L/b evenly spaced, overlapping blocks of length b covering 1..L.
n = 1;
if b < L
    n = ceil(1.5 * L / b);
end
starts = round(linspace(1, L - b + 1, n));
centres = starts + (b - 1) / 2;
end

function NR = blockReferences(NR, refImg, smoothSigma)
% Template spectrum, taper and offset for every block, stacked on dim 3.
[NR.maskMul, NR.maskOffset] = deal(zeros([NR.bs NR.nB], 'like', refImg));
NR.cfRef = complex(NR.maskMul);
for j = 1:NR.nB
    Rj = makeRef(refImg(NR.rows(j, :), NR.cols(j, :)), smoothSigma);
    NR.maskMul(:, :, j) = Rj.maskMul;
    NR.maskOffset(:, :, j) = Rj.maskOffset;
    NR.cfRef(:, :, j) = Rj.cfRef;
end
end

function [dy, dx, low] = blockPhaseCorr(Y, NR, subpixel)
% Phase correlation of every block of every (rigidly aligned) frame in Y
% against the matching template block. Returns [nBlocks x n] shifts and
% the blocks that were smoothed because of low SNR.
n = size(Y, 3);
nB = NR.nB;
l = NR.lcorr;
[dy, dx] = deal(zeros(nB, n));
low = false(nB, n);
for s = 1:NR.subBatch:n
    fi = s:min(s + NR.subBatch - 1, n);
    m = numel(fi);
    Yb = zeros([NR.bs nB m], 'like', Y);        % blocks: [bh bw nB m]
    for j = 1:nB
        Yb(:, :, j, :) = reshape(Y(NR.rows(j, :), NR.cols(j, :), fi), [NR.bs 1 m]);
    end
    cc = whitenedXcorr(Yb, NR.maskMul, NR.maskOffset, NR.cfRef);
    clear Yb
    cc = reshape(cc(lagIdx(NR.bs(1), l), lagIdx(NR.bs(2), l), :, :), (2*l+1)^2, nB, m);
    [cc, low(:, fi)] = smoothLowSNR(cc, NR);
    [py, px] = peakSubpixel(reshape(cc, [], nB * m), l, subpixel);
    dy(:, fi) = reshape(py, nB, m);
    dx(:, fi) = reshape(px, nB, m);
end
end

function [cc, lowFirst] = smoothLowSNR(cc, NR)
% cc: [L*L x nB x m]. Blocks whose peak-to-sidelobe ratio is below
% NR.snrThresh get the Gaussian-weighted average of their neighbours'
% correlation maps (up to 2 rounds, as in suite2p).
[L2, nB, m] = size(cc);
L = 2 * NR.lcorr + 1;
lagY = cast(mod((0:L2-1)', L) - NR.lcorr, 'like', cc);
lagX = cast(floor((0:L2-1)' / L) - NR.lcorr, 'like', cc);
Wt = cast(NR.Wn.', 'like', cc);
for round_ = 1:2
    low = blockSNR(cc, lagY, lagX) < NR.snrThresh;
    if round_ == 1
        lowFirst = gather(low);
    end
    if ~any(low(:))
        break
    end
    % neighbour average: ccS(:, j, t) = sum_j' Wn(j, j') * cc(:, j', t)
    ccS = reshape(permute(reshape(reshape(permute(cc, [1 3 2]), L2 * m, nB) * Wt, L2, m, nB), [1 3 2]), L2, []);
    cc = reshape(cc, L2, []);
    cc(:, low(:)) = ccS(:, low(:));
    cc = reshape(cc, L2, nB, m);
end
end

function snr = blockSNR(cc, lagY, lagX)
% Peak / highest value more than 2 px (in y or x) away from the peak.
% cc: [L*L x nB x m]; returns [nB x m].
[pk, k] = max(cc, [], 1);
side = cc;
% reshape: lagY(k) takes the (column) orientation of lagY when k is a vector,
% i.e. when a sub-batch holds a single frame (m = 1) or there is one block
lagYk = reshape(lagY(k), size(k));
lagXk = reshape(lagX(k), size(k));
side(abs(lagY - lagYk) <= 2 & abs(lagX - lagXk) <= 2) = -Inf;
side = max(side, [], 1);
snr = pk ./ side;
snr(side <= 0) = Inf;
snr = reshape(snr, size(cc, 2), size(cc, 3));
end

function Z = warpFrames(Y, dy, dx, bdy, bdx, NR, method)
% corrected(y,x) = Y(y + dy + DY(y,x), x + dx + DX(y,x)) on the output grid,
% where DY/DX interpolate the block shifts bilinearly between block centres.
[H, W, n] = size(Y);
Z = zeros([size(NR.Yo) n], 'like', Y);
bdy = cast(bdy, 'like', Y);
bdx = cast(bdx, 'like', Y);
field = @(bd) NR.Ay * reshape(bd, NR.nby, NR.nbx) * NR.Ax.';
for k = 1:n
    Yq = min(max(NR.Yo + dy(k) + field(bdy(:, k)), 1), H);    % clamp = replicate edges
    Xq = min(max(NR.Xo + dx(k) + field(bdx(:, k)), 1), W);
    Z(:, :, k) = interp2(Y(:, :, k), Xq, Yq, method);
end
end

function M = linInterpMatrix(q, centres)
% [numel(q) x numel(centres)] matrix of 1-D linear interpolation weights
% from values at 'centres' (ascending) to query points q. Queries outside
% the first/last centre take the edge value. Rows sum to 1.
M = ones(numel(q), 1);
if numel(centres) == 1
    return
end
M = interp1(centres(:), eye(numel(centres)), min(max(q(:), centres(1)), centres(end)), 'linear');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PASS 2 BATCH & PARALLEL HELPERS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Z, bdy, bdx, blow, dg] = pass2Batch(io, gi, dy, dx, s)
% One batch of pass 2 (runs on a worker or here): optional non-rigid block
% estimate, then shift/warp the output channel to uint16. Z = [] when not
% writing; bdy/bdx/blow = [] when rigid only; dg = [] unless s.diag (batch
% statistics for the diagnostic figures).
NR = s.NR;
if isa(NR, 'parallel.pool.Constant')
    NR = NR.Value;
end
[Z, bdy, bdx, blow, dg] = deal([]);
if s.nonrigid
    % rigidly align the estimation region, then block phase correlation
    Ya = loadFrames(io, gi, s.alignChannel);
    Yr = applyShifts(Ya(s.er, s.ec, :), dy, dx, 1:s.He, 1:s.We);
    [bdy, bdx, blow] = blockPhaseCorr(Yr, NR, s.subpixel);
    clear Yr
end
if s.writeOutput
    if s.nonrigid && s.channel == s.alignChannel
        Y = Ya;
    else
        Y = loadFrames(io, gi, s.channel);
    end
    if s.nonrigid
        Z = warpFrames(Y, dy, dx, bdy, bdx, NR, s.warpMethod);
    else
        Z = applyShifts(Y, dy, dx, s.outRows, s.outCols);
    end
    Z = uint16(gather(Z));
    if ~isempty(s.diag)
        dg = batchDiag(Y, Z, gi, s.outRows, s.outCols, s.diag);
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DIAGNOSTICS COLLECTED DURING PASS 2    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dgn, dspec] = initDiag(p, N, outRows, outCols, er, ec, refImg)
% Collectors (dgn, kept here) and what each batch needs (dspec).
ns = min(optGet(p.diagOpts, 'nSample', 1000), N);
nBins = optGet(p.diagOpts, 'nBins', 6);
ly = numel(outRows);
lx = numel(outCols);
rI = intersect(outRows, er);               % pixels shared by the output and the template
cI = intersect(outCols, ec);
dspec = struct('sample', unique(round(linspace(1, N, ns))), 'nBins', nBins, 'N', N, ...
    'zR', rI - outRows(1) + 1, 'zC', cI - outCols(1) + 1, ...
    'ref', single(refImg(rI - er(1) + 1, cI - ec(1) + 1)));
ns = numel(dspec.sample);
dgn = struct('sampleIdx', dspec.sample, 'nBins', nBins, ...
    'meanFraw', zeros(N, 1), 'meanFcor', zeros(N, 1), 'cRaw', zeros(N, 1), 'cCor', zeros(N, 1), ...
    'sumRaw', zeros(ly, lx, nBins), 'sumCor', zeros(ly, lx, nBins), 'cnt', zeros(1, nBins), ...
    'Xraw', zeros(ly, lx, ns, 'uint16'), 'Xcor', zeros(ly, lx, ns, 'uint16'));
end

function dg = batchDiag(Y, Z, gi, outRows, outCols, dspec)
% Statistics of one batch: Y = raw frames of the output channel (after the
% scan-phase correction, before motion correction), Z = corrected frames
% (uint16), compared on the same output pixels.
Yo = uint16(gather(Y(outRows, outCols, :)));
n = numel(gi);
dg.meanFraw = reshape(double(mean(mean(Yo, 1), 2)), [], 1);
dg.meanFcor = reshape(double(mean(mean(Z, 1), 2)), [], 1);
% correlation of every frame with the registration template (shared pixels)
ref = dspec.ref(:) - mean(dspec.ref(:));
ref = ref / norm(ref);
dg.cRaw = zeros(n, 1);
dg.cCor = zeros(n, 1);
for k0 = 1:100:n
    k = k0:min(k0 + 99, n);
    A = single(reshape(Yo(dspec.zR, dspec.zC, k), [], numel(k)));
    A = A - mean(A, 1);
    dg.cRaw(k) = double((ref' * A) ./ sqrt(sum(A.^2, 1)))';
    B = single(reshape(Z(dspec.zR, dspec.zC, k), [], numel(k)));
    B = B - mean(B, 1);
    dg.cCor(k) = double((ref' * B) ./ sqrt(sum(B.^2, 1)))';
end
% sums for the time-bin mean images
bin = min(dspec.nBins, ceil(gi / dspec.N * dspec.nBins));
ub = unique(bin(:))';
dg.bins = ub;
dg.cnt = arrayfun(@(b) nnz(bin == b), ub);
dg.sumRaw = zeros(size(Yo, 1), size(Yo, 2), numel(ub));
dg.sumCor = zeros(size(Yo, 1), size(Yo, 2), numel(ub));
for i = 1:numel(ub)
    dg.sumRaw(:, :, i) = sum(Yo(:, :, bin == ub(i)), 3, 'double');
    dg.sumCor(:, :, i) = sum(Z(:, :, bin == ub(i)), 3, 'double');
end
% PCA sample frames
sel = ismember(gi, dspec.sample);
dg.sIdx = gi(sel);
dg.Xraw = Yo(:, :, sel);
dg.Xcor = Z(:, :, sel);
end

function dgn = collectDiag(dgn, gi, dg)
dgn.meanFraw(gi) = dg.meanFraw;
dgn.meanFcor(gi) = dg.meanFcor;
dgn.cRaw(gi) = dg.cRaw;
dgn.cCor(gi) = dg.cCor;
for i = 1:numel(dg.bins)
    b = dg.bins(i);
    dgn.sumRaw(:, :, b) = dgn.sumRaw(:, :, b) + dg.sumRaw(:, :, i);
    dgn.sumCor(:, :, b) = dgn.sumCor(:, :, b) + dg.sumCor(:, :, i);
    dgn.cnt(b) = dgn.cnt(b) + dg.cnt(i);
end
if ~isempty(dg.sIdx)
    [~, pos] = ismember(dg.sIdx, dgn.sampleIdx);
    dgn.Xraw(:, :, pos) = dg.Xraw;
    dgn.Xcor(:, :, pos) = dg.Xcor;
end
end

function v = optGet(c, name, default)
% value of a name-value pair in the cell c, or default
v = default;
k = find(strcmpi(c(1:2:end), name), 1);
if ~isempty(k)
    v = c{2*k};
end
end

function pool = startPool(nWorkers, profile)
% The open pool (any kind), or a new one from 'profile' (the default
% profile if empty); [] if none can be started.
pool = [];
try
    pool = gcp('nocreate');
    if isempty(pool)
        args = {};
        if ~isempty(profile)
            args{end+1} = profile;
        end
        if ~isempty(nWorkers)
            args{end+1} = nWorkers;
        end
        pool = parpool(args{:});
    end
catch ME
    fprintf('No parallel pool (%s); running serially.\n', ME.message);
    pool = [];
end
end

function nW = poolWorks(pool, label, fn, varargin)
% Run fn(varargin{:}) once on a worker. Returns the pool size if it works,
% or 0 if there is no pool or it errors (e.g. a function that thread-based
% workers do not support).
nW = 0;
if isempty(pool)
    return
end
try
    fetchOutputs(parfeval(pool, fn, 0, varargin{:}));
    nW = pool.NumWorkers;
catch ME
    fprintf('Parallel pool cannot run %s (%s); running it serially.\n', label, ME.message);
end
end

function probePass1(io, gi, channel, er, ec, R, lcorr, subpixel)
% Pass 1 on a few frames, to test the pool before the real run.
Y = loadFrames(io, gi, channel);
phaseCorr(Y(er, ec, :), R, lcorr, subpixel);
end

function q = progressQueue(verbose, nW, label, n)
% Progress reporter usable inside parfor: [] (silent), a struct (serial:
% prints directly) or a DataQueue (parallel: prints on this MATLAB).
q = [];
if ~verbose
    return
end
if nW == 0
    q = struct('label', label, 'n', n);
else
    q = parallel.pool.DataQueue;
    afterEach(q, @(b) fprintf('  %s: batch %d/%d done\n', label, b, n));
end
end

function reportProgress(q, b)
if isstruct(q)
    fprintf('  %s: batch %d/%d\n', q.label, b, q.n);
elseif ~isempty(q)
    send(q, b);
end
end


%%%%%%%%%%%%%%%%%%%
%%% I/O HELPERS %%%
%%%%%%%%%%%%%%%%%%%

function Y = loadFrames(io, gidx, channel)
% Frames with global indices gidx from one channel, cropped, as single
% (on the GPU if io.useGPU).
Y = readFrames(io.src, io.cum, gidx, channel, io.rc, io.bidi);
if io.useGPU
    Y = gpuArray(Y);      % move uint16 first: half the transfer size
end
Y = single(Y);
end

function Y = toDevice(Y, useGPU)
if useGPU
    Y = gpuArray(Y);
end
Y = single(Y);
end

function Y = readFrames(src, cum, gidx, channel, rc, bidi)
% Read frames with global (concatenated, 1-based) indices gidx, sorted
% ascending, from one PMT channel; returns cropped uint16 [rows cols n].
% bidi = [shiftOdd shiftEven] px: column shift of the odd / even raw rows
% (out(:,c) = in(:,c-shift)) applied before cropping; [0 0] = none.
% The file reading and the scan-phase (even/odd row) shift are in
% sbx_method.read, shared with the diagnostics.
if nargin < 6
    bidi = [0 0];
end
Y = zeros(rc(3)-rc(1)+1, rc(4)-rc(2)+1, numel(gidx), 'uint16');
for f = 1:numel(src)
    sel = find(gidx > cum(f) & gidx <= cum(f+1));
    if isempty(sel)
        continue
    end
    d = src(f).first + gidx(sel) - cum(f) - 1;    % 0-based frame on disk
    Y(:, :, sel) = sbx_method.read(src(f).path, d, 'channel', channel, 'crop', rc, ...
        'bidi', bidi, 'info', src(f));
end
end

function tw = openTiff(path)
% BigTIFF for writing, creating the folder if needed.
d = fileparts(path);
if ~isempty(d) && ~isfolder(d)
    mkdir(d);
end
tw = Tiff(path, 'w8');
end

function t = tiffTags(h, w)
t = struct('ImageLength', h, 'ImageWidth', w, ...
    'Photometric', Tiff.Photometric.MinIsBlack, 'BitsPerSample', 16, ...
    'SamplesPerPixel', 1, 'SampleFormat', Tiff.SampleFormat.UInt, ...
    'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky, ...
    'Compression', Tiff.Compression.None, 'RowsPerStrip', h);
end

function n = fastFFTSize(n)
% Largest m <= n whose prime factors are all in {2,3,5,7}.
while any(factor(n) > 7)
    n = n - 1;
end
end

function v = prctileBase(x, q)
% Percentile without the Statistics Toolbox (linear interpolation).
x = sort(x(:));
v = x(1);
if numel(x) > 1
    v = interp1(linspace(0, 100, numel(x)), x, q);
end
end
