classdef mc_diag_method
    % MC_DIAG_METHOD  Static helpers to check motion correction and registration.
    % Call as mc_diag_method.<method>(...); no object is needed. File reading
    % goes through sbx_method.
    %
    %   mc_diag_method.session(mcTifOrData, sbxFiles, outDir, ...)  figures 01-08 + summary
    %        (also what sbx_motion_correct calls with 'diagnostics', true)
    %   mc_diag_method.segments(sbxFiles, outPng, ...)              raw mean / max of each segment
    %   mc_diag_method.referenceFigure(refTif, mcPath, outPng, name)  reference-to-session figure
    %   M = mc_diag_method.metrics(mov, ...)                        PC metric, frame correlation, kymographs
    %   [dy, dx, peak] = mc_diag_method.shift(ref, img)             phase-correlation translation

    methods (Static)

        function M = metrics(mov, varargin)
        % METRICS  Suite2p-style PC-based motion-correction quality metrics.
        %
        % Implements "Registration metrics" from Stringer & Pachitariu,
        % "Extracting large-scale neural activity with Suite2p" (bioRxiv 2026):
        %   1. subsample 2000-5000 registered frames, crop to valid region
        %   2. PCA -> top nPC components
        %   3. for each PC: mean of nTop frames with largest / smallest projection
        %   4. register bottom-mean to top-mean (rigid, then non-rigid on blocks)
        %   5. report offset lengths: rigid, mean NR, max NR, mean(rigid+NR)
        % Plus two more direct checks, on every frame of the (cropped) movie:
        %   6. frame-to-mean correlation (as in CaImAn/NoRMCorre): Pearson
        %      correlation of each frame with the mean image. Higher and flatter
        %      after correction is better; remaining dips = residual motion.
        %   7. kymographs: one horizontal and one vertical line through the mean
        %      image (auto-picked where the high-pass mean image has the most edges,
        %      or set with kymoRow / kymoCol). The band of kymoWidth px around each
        %      line is averaged into one profile per frame, and profiles are averaged
        %      over kymoBin frames. Straight stripes over time = stable; stripes that
        %      bend together = residual motion; stripes that fade without moving =
        %      likely z-motion.
        % If 'rawMov' (the movie before correction, same pixels) is given, 6 and 7
        % are also computed for it, on the same crop and lines, for before/after
        % (and 1-5 too if 'rawPC' is true).
        %
        % Movies too large for memory: pass the tif path instead of the array. The
        % movie is then read in passes and never held whole:
        %   - sampling pass: only the 2000-5000 PCA sample frames are read (kept in
        %     their native class, e.g. uint16: ~3.4 GB for 5000 frames of 493x692).
        %     Their mean is used as the mean image for 6 and 7 (evenly spaced
        %     frames, so it matches the full mean closely). PCA uses the
        %     frames x frames Gram matrix, built over blocks of pixels.
        %   - streaming pass: chunk frames at a time for 6 and 7.
        % Counting the frames of a tif walks its page headers once (fast, no image
        % data). An in-memory array is processed the same way.
        %
        %   M = mc_diag_method.metrics(mov)
        %   M = mc_diag_method.metrics(mov, 'nPC',30, 'nTop',300, 'blockSize',[128 128], ...
        %                            'yrange',[], 'xrange',[], 'smoothSigma',1.15, ...
        %                            'maxShiftFrac',0.1, 'maxShiftNR',5, 'plot',true, ...
        %                            'rawMov',[], 'rawPC',false, 'kymoRow',[], 'kymoCol',[], ...
        %                            'kymoWidth',5, 'kymoBin',5, 'hpSigma',10, ...
        %                            'maxFrames',Inf, 'chunk',200)
        %
        %   mov        registered movie: Ly x Lx x T array, a tif path (e.g.
        %              ConcatenatedMovie_MC.tif), or a struct with field 'path' and
        %              optional 'rows', 'cols' (only those pixels are used)
        %   rawMov     movie before correction ([] = skip), same forms as mov. It
        %              must cover the same pixels as mov. For sbx_motion_correct
        %              output and a raw tif exported with the same 'crop', use
        %              struct('path', rawTif, 'rows', S.outRows, 'cols', S.outCols)
        %              with S from the _shifts.mat. Its frame count may differ from
        %              mov (e.g. first frame of each segment not dropped).
        %   rawPC      also run the PC metric (1-5) on rawMov
        %   kymoRow    row (in the yrange/xrange crop) of the horizontal kymograph
        %              line ([] = auto)
        %   kymoCol    column (in the crop) of the vertical line ([] = auto)
        %   kymoWidth  band width (px) averaged across the line
        %   kymoBin    frames averaged per kymograph time bin
        %   hpSigma    Gaussian sigma (px) of the high-pass filter used to auto-pick
        %              the lines
        %   maxFrames  use only the first maxFrames frames of each movie
        %   chunk      frames per read in the streaming pass
        %   nSample    frames for the PCA sample ([] = 2000 or 5000 as described above)
        %   streamStride  streaming pass (frame correlation, mean F, kymographs, time
        %              bins) uses every Nth frame (1 = all)
        %   nBins      >0: also mean images of nBins consecutive time bins
        %              (M.binImg), e.g. for first-vs-last comparisons
        %
        % A movie / rawMov may also be a Scanbox recording read directly, no tif:
        %   struct('sbx', <path or cell of paths, concatenated in order>, ...
        %          'channel', 1, 'crop', [1 80 Inf Inf], 'first', 1, 'bidi', [0 0], ...
        %          'rows', [], 'cols', [])
        % 'first' = frames dropped at the start of every file (1 = as
        % sbx_motion_correct), 'rows'/'cols' select pixels of the cropped frame
        % (e.g. S.outRows / S.outCols of the _shifts.mat to match the corrected movie).
        %
        % Output struct M:
        %   regDX     nPC x 4  [rigid, meanNR, maxNR, mean(rigid+NR)] offset lengths (px)
        %   tPC       nFrames x nPC   PC time courses
        %   topMean   ly x lx x nPC   mean of top frames per PC (cropped)
        %   botMean   ly x lx x nPC   mean of bottom frames per PC
        %   frameIdx  indices of sampled frames
        %   meanImg   mean image of the cropped movie (mean of the sampled frames)
        %   frameCorr nStream x 1  correlation of each streamed frame with meanImg
        %   streamIdx frame numbers of the streamed frames (1:streamStride:T)
        %   meanF     nStream x 1  mean fluorescence of each streamed frame (cropped)
        %   binImg    ly x lx x nBins mean image per time bin ([] if nBins = 0)
        %   movSize   [Ly Lx T] of mov (after maxFrames)
        %   frameCorrRaw, meanImgRaw, rawSize   same for rawMov ([] if not given)
        %   regDXRaw, tPCRaw, topMeanRaw, botMeanRaw, frameIdxRaw   PC metric of
        %             rawMov ([] unless rawPC)
        %   kymo      struct: row, col (line positions in the crop), width, bin,
        %             t (center frame of each time bin), H (lx x nBins, along the
        %             horizontal line), V (ly x nBins, along the vertical line),
        %             tRaw, Hraw, Vraw (same for rawMov, [] if not given)

        p = inputParser;
        p.addParameter('nPC', 30);
        p.addParameter('nTop', 300);
        p.addParameter('blockSize', [128 128]);
        p.addParameter('yrange', []);
        p.addParameter('xrange', []);
        p.addParameter('smoothSigma', 1.15);
        p.addParameter('maxShiftFrac', 0.1);
        p.addParameter('maxShiftNR', 5);
        p.addParameter('plot', true);
        p.addParameter('rawMov', []);
        p.addParameter('rawPC', false);
        p.addParameter('kymoRow', []);
        p.addParameter('kymoCol', []);
        p.addParameter('kymoWidth', 5);
        p.addParameter('kymoBin', 5);
        p.addParameter('hpSigma', 10);
        p.addParameter('maxFrames', Inf);
        p.addParameter('chunk', 200);
        p.addParameter('nSample', []);
        p.addParameter('streamStride', 1);
        p.addParameter('nBins', 0);
        p.parse(varargin{:});
        o = p.Results;
        rawMov = o.rawMov;
        o.rawMov = [];      % keep the movie out of M.opts

        src = open_source(mov, o.maxFrames, 'movie');
        [Ly, Lx, T] = deal(src.Ly, src.Lx, src.T);
        raw = [];
        if ~isempty(rawMov)
            raw = open_source(rawMov, o.maxFrames, 'raw movie');
            if ~isequal([raw.Ly raw.Lx], [Ly Lx])
                error(['rawMov is %dx%d px but mov is %dx%d px; give the pixels that match ' ...
                    '(e.g. struct(''path'', rawTif, ''rows'', S.outRows, ''cols'', S.outCols) ' ...
                    'with S from the _shifts.mat).'], raw.Ly, raw.Lx, Ly, Lx);
            end
        end

        % ---- crop to valid region
        if isempty(o.yrange), o.yrange = [1 Ly]; end
        if isempty(o.xrange), o.xrange = [1 Lx]; end
        yr = o.yrange(1):o.yrange(2);
        xr = o.xrange(1):o.xrange(2);
        [ly, lx] = deal(numel(yr), numel(xr));

        % ---- 1-5. PC metric on the sampled frames (sample mean = mean image)
        frameIdx = sample_index(T, Ly, Lx, o.nSample);
        fprintf('  reading %d sample frames ...\n', numel(frameIdx));
        X = read_frames(src, frameIdx, yr, xr);
        meanImg = mean_image(X);
        [regDX, tPC, topMean, botMean] = pc_metric(X, o);
        clear X

        [frameCorrRaw, meanImgRaw, frameIdxRaw] = deal([]);
        [regDXRaw, tPCRaw, topMeanRaw, botMeanRaw] = deal([]);
        if ~isempty(raw)
            frameIdxRaw = sample_index(raw.T, Ly, Lx, o.nSample);
            fprintf('  reading %d raw sample frames ...\n', numel(frameIdxRaw));
            Xr = read_frames(raw, frameIdxRaw, yr, xr);
            meanImgRaw = mean_image(Xr);
            if o.rawPC
                [regDXRaw, tPCRaw, topMeanRaw, botMeanRaw] = pc_metric(Xr, o);
            end
            clear Xr
        end

        % ---- 7. kymograph lines (picked on the mean image)
        h = floor(o.kymoWidth / 2);
        [kRow, kCol] = deal(o.kymoRow, o.kymoCol);
        if isempty(kRow) || isempty(kCol)
            [aRow, aCol] = pick_kymo_lines(meanImg, h, o.hpSigma);
            if isempty(kRow), kRow = aRow; end
            if isempty(kCol), kCol = aCol; end
        end
        kRow = min(max(round(kRow), 1 + h), ly - h);
        kCol = min(max(round(kCol), 1 + h), lx - h);

        % ---- 6-7. streaming pass: frame-to-mean correlation and line profiles
        fprintf('  streaming %d frames (every %d) ...\n', T, o.streamStride);
        [frameCorr, H, V, meanF, streamIdx, binImg] = stream_pass(src, yr, xr, meanImg, kRow, kCol, h, ...
            o.chunk, o.streamStride, o.nBins);
        kymo = struct('row', kRow, 'col', kCol, 'width', 2*h + 1, 'bin', o.kymoBin);
        [kymo.H, kymo.V, kymo.t] = bin_profiles(H, V, o.kymoBin);
        kymo.t = streamIdx(1) + (kymo.t - 1) * o.streamStride;      % bin centres in frame numbers
        [kymo.Hraw, kymo.Vraw, kymo.tRaw] = deal([]);
        [meanFRaw, streamIdxRaw, binImgRaw] = deal([]);
        if ~isempty(raw)
            fprintf('  streaming %d raw frames (every %d) ...\n', raw.T, o.streamStride);
            [frameCorrRaw, H, V, meanFRaw, streamIdxRaw, binImgRaw] = stream_pass(raw, yr, xr, meanImgRaw, ...
                kRow, kCol, h, o.chunk, o.streamStride, o.nBins);
            [kymo.Hraw, kymo.Vraw, kymo.tRaw] = bin_profiles(H, V, o.kymoBin);
            kymo.tRaw = streamIdxRaw(1) + (kymo.tRaw - 1) * o.streamStride;
        end
        clear H V

        rawSize = [];
        if ~isempty(raw), rawSize = [raw.Ly raw.Lx raw.T]; end
        M = struct('regDX', regDX, 'tPC', tPC, 'topMean', topMean, ...
                   'botMean', botMean, 'frameIdx', frameIdx, 'meanImg', meanImg, ...
                   'frameCorr', frameCorr, 'movSize', [Ly Lx T], ...
                   'frameCorrRaw', frameCorrRaw, 'meanImgRaw', meanImgRaw, 'rawSize', rawSize, ...
                   'regDXRaw', regDXRaw, 'tPCRaw', tPCRaw, 'topMeanRaw', topMeanRaw, ...
                   'botMeanRaw', botMeanRaw, 'frameIdxRaw', frameIdxRaw, ...
                   'streamIdx', streamIdx, 'meanF', meanF, 'binImg', binImg, ...
                   'streamIdxRaw', streamIdxRaw, 'meanFRaw', meanFRaw, 'binImgRaw', binImgRaw, ...
                   'kymo', kymo, 'opts', o);

        if o.plot
            plot_mc_diag_method_metrics(M);
            plot_frame_corr(M);
            plot_kymographs(M);
        end
        end

        function D = session(src, sbxFiles, outDir, varargin)
        % SESSION  Motion-correction diagnostic figures for one session.
        % Two input modes, one set of figures:
        %
        %   PATHS      read the motion-corrected tif and the raw .sbx files (no tif
        %              conversion of the raw data):
        %                D = mc_diag_method.session(mcPath, sbxFiles, outDir, ...)
        %   VARIABLES  plot from data already in memory (what sbx_motion_correct
        %              hands over with 'diagnostics', true; nothing is re-read):
        %                D = mc_diag_method.session(data, [], outDir, ...)
        %
        %   D = mc_diag_method.session(mcPath, sbxFiles, outDir, 'name', 'session', ...
        %                      'channel', 1, 'nPC', 30, 'segNPC', 5, 'nSample', 2000, ...
        %                      'stride', 5, 'nBins', 6, 'zoomSize', 120, 'smoothWin', 50)
        %
        %   src       ConcatenatedMovie_MC.tif (its _shifts.mat is read too), or the
        %             'data' struct described below
        %   sbxFiles  cell of the .sbx files (paths, extension optional) that were
        %             concatenated, in the same order as in the motion correction
        %             (paths mode only; [] otherwise)
        %   outDir    folder for the figures (created)
        %
        %   data struct (variables mode):
        %     corSample, rawSample  [ly lx ns] uint16: the same ns evenly spaced
        %                           frames, corrected / before motion correction,
        %                           same output pixels
        %     sampleIdx             frame numbers of those samples
        %     meanF, frameCorr      structs with fields cor, raw: N x 1 per frame
        %     corrRef               what frameCorr is relative to ('template' or
        %                           'mean image')
        %     binImg                struct cor, raw: ly x lx x nBins mean images
        %     shifts                struct with the _shifts.mat fields dy, dx, cmax,
        %                           bad, nonrigid, frameIndex, crop, bidi, outRows,
        %                           outCols, params
        %     segNames              cell, one name per concatenated segment
        %     rawNote               how 'raw' was made (shown in titles)
        %   name      label used in figure titles
        %   channel   PMT channel of the corrected movie (1 = green)
        %   nPC       PCs of the corrected movie for the registration-offset metric
        %   segNPC    PCs for each raw segment (only PC1 is plotted)
        %   nSample   frames sampled for PCA (per movie / segment)
        %   stride    every Nth frame in the streaming pass (frame correlation, mean
        %             F, kymographs, time bins)
        %   nBins     time bins for the early-vs-late comparison
        %   zoomSize  side (px) of the zoomed patches
        %   smoothWin smoothing window (streamed samples) for the time-course plots
        %
        % Raw movies are the .sbx frames cropped with the same raw crop and output
        % rectangle as the corrected movie (crop.raw, outRows, outCols of the
        % _shifts.mat), WITHOUT motion or scan-phase correction, so raw and
        % corrected cover the same pixels.
        %
        % Figures (PNG, in outDir):
        %   01_timecourses            mean F per frame, rigid dy/dx, non-rigid
        %                             residual, correlation peak, frame-to-mean
        %                             correlation (raw vs corrected), PC1
        %   02_PC1_corrected          PC1 top / bottom means and their difference,
        %                             PC offsets (Suite2p registration metric)
        %   03_PC1_raw_segments       the same PC1 maps for every raw segment
        %   04_mean_image_raw_vs_corrected   mean images and their difference
        %   05_zoom_raw_vs_corrected  zoomed patches (brightest, most structured)
        %   06_bidi_scan_phase        even/odd scan-line offset: estimate, applied,
        %                             residual (needs out.bidi in the _shifts.mat)
        %   07_early_vs_late          first vs last time bin, raw and corrected
        %   08_kymographs             line profiles over time, raw and corrected
        % plus mc_diagnostics_summary.txt and mc_diagnostics.mat.

        ip = inputParser;
        ip.addParameter('name', '');
        ip.addParameter('channel', 1);
        ip.addParameter('nPC', 10);
        ip.addParameter('segNPC', 5);
        ip.addParameter('nSample', 2000);
        ip.addParameter('stride', 5);
        ip.addParameter('nBins', 6);
        ip.addParameter('zoomSize', 120);
        ip.addParameter('smoothWin', 50);
        ip.parse(varargin{:});
        o = ip.Results;

        % gather the metrics: by reading the files, or from variables handed over by
        % sbx_motion_correct (same M / Seg / S contents, same plots below)
        if isstruct(src)
            if isempty(o.name), o.name = 'session'; end
            [M, Seg, S, segNames, lbl] = fromData(src, o);
        else
            if isempty(o.name), [~, o.name] = fileparts(src); end
            [M, Seg, S, segNames, lbl] = fromFiles(src, sbxFiles, o);
        end
        if ~isfolder(outDir), mkdir(outDir); end
        ttl = strrep(o.name, '_', '\_');
        nSeg = numel(segNames);
        N = numel(S.dy);
        if N == 0, N = M.movSize(3); end
        segBreaks = [];
        if ~isempty(S.frameIndex)
            segBreaks = find(diff(S.frameIndex(:, 1)) ~= 0) + 1;
        end
        segStart = [0 cumsum([Seg.T])];

        %% derived numbers
        nS = numel(M.streamIdx);
        b1 = 1:max(1, round(nS / o.nBins));                  % first time bin (streamed samples)
        F0 = mean(M.meanF(b1));
        F0raw = mean(M.meanFRaw(b1));
        shp.raw = sharpness(M.meanImgRaw);
        shp.cor = sharpness(M.meanImg);
        cs = @(x) [median(x(~isnan(x))), prctile1(x(~isnan(x)), 1)];
        fcCor = cs(M.frameCorr);
        fcRaw = cs(M.frameCorrRaw);
        nr = [];
        if ~isempty(S.nonrigid) && isstruct(S.nonrigid)
            L = hypot(double(S.nonrigid.dy), double(S.nonrigid.dx));
            nr = struct('mean', mean(L, 2), 'max', max(L, [], 2));
        end
        % early vs late (corrected and raw)
        el = struct();
        for kind = {'cor', 'raw'}
            B = M.binImg;
            if strcmp(kind{1}, 'raw'), B = M.binImgRaw; end
            a = double(B(:, :, 1));
            z = double(B(:, :, end));
            [gx, gy] = imgradientxy(0.5 * (a + z), 'central');
            ctr = {round(size(a, 1)*0.1):round(size(a, 1)*0.9), round(size(a, 2)*0.1):round(size(a, 2)*0.9)};
            d = z - a;
            [sy, sx] = mc_diag_method.shift(a, z);
            el.(kind{1}) = struct('first', a, 'last', z, 'diff', d, 'dy', sy, 'dx', sx, ...
                'corrGain', corr2(d(ctr{:}), 0.5 * (a(ctr{:}) + z(ctr{:}))), ...
                'corrShift', max(abs([corr2(d(ctr{:}), gx(ctr{:})), corr2(d(ctr{:}), gy(ctr{:}))])));
            el.(kind{1}).sharpPerBin = arrayfun(@(b) sharpness(B(:, :, b)), 1:size(B, 3));
        end
        binCorr = nan(o.nBins);
        for a = 1:o.nBins
            for b = 1:o.nBins
                x = M.binImg(:, :, a); y = M.binImg(:, :, b);
                binCorr(a, b) = corr2(x, y);
            end
        end

        %% FIG 1: time courses
        cRaw = [0.85 0.33 0.10];
        f = newfig([1100 1250]);
        tl = tiledlayout(6, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
        ax = gobjects(6, 1);
        ax(1) = nexttile; hold on
        plot(M.streamIdxRaw, M.meanFRaw / F0raw, 'Color', [1 0.8 0.65]);
        plot(M.streamIdx, M.meanF / F0, 'Color', [0.7 0.7 0.7]);
        hR = plot(M.streamIdxRaw, movmean(M.meanFRaw / F0raw, o.smoothWin), 'Color', cRaw, 'LineWidth', 1.3);
        hC = plot(M.streamIdx, movmean(M.meanF / F0, o.smoothWin), 'k', 'LineWidth', 1.3);
        legend([hR hC], {'raw (smoothed)', 'corrected (smoothed)'}, 'Location', 'best', 'Box', 'off');
        ylabel('mean F / first time bin'); title('Mean fluorescence per frame (dashed lines = segment boundaries)');
        ax(2) = nexttile; hold on
        if ~isempty(S.dy)
            hy = plot(1:N, S.dy, 'b'); hx = plot(1:N, S.dx, 'r');
            legend([hy hx], {'dy (rows)', 'dx (columns)'}, 'Location', 'best', 'Box', 'off');
        end
        ylabel('rigid shift (px)'); title(sprintf('Rigid shift applied by motion correction  (SD: dy %.2f, dx %.2f px)', ...
            std(S.dy), std(S.dx)));
        ax(3) = nexttile; hold on
        if ~isempty(nr)
            h1 = plot(1:N, nr.max, 'Color', [0.75 0.75 0.75]);
            h2 = plot(1:N, nr.mean, 'k');
            legend([h2 h1], {'mean over blocks', 'max over blocks'}, 'Location', 'best', 'Box', 'off');
        else
            text(0.5, 0.5, 'non-rigid correction off', 'Units', 'normalized', 'HorizontalAlignment', 'center');
        end
        ylabel('residual block shift (px)'); title('Non-rigid residual shift after the rigid step (per frame, over blocks)');
        ax(4) = nexttile; hold on
        if ~isempty(S.cmax)
            plot(1:N, S.cmax, 'k');
            if ~isempty(S.bad) && any(S.bad)
                plot(find(S.bad), S.cmax(S.bad), 'r.');
            end
        end
        ylabel('correlation peak'); title(sprintf('Phase-correlation peak of the rigid fit (red = outlier frames, %.2f %%)', ...
            100 * mean(S.bad)));
        ax(5) = nexttile; hold on
        plot(M.streamIdxRaw, M.frameCorrRaw, 'Color', cRaw);
        plot(M.streamIdx, M.frameCorr, 'k');
        legend({sprintf('raw (median %.3f)', fcRaw(1)), sprintf('corrected (median %.3f)', fcCor(1))}, ...
            'Location', 'best', 'Box', 'off');
        ylabel(['corr. with ' lbl.ref]); title(sprintf('Frame-to-%s correlation (dips = residual motion, z-change or blur)', lbl.ref));
        ax(6) = nexttile; hold on
        plot(M.frameIdx, M.tPC(:, 1), 'k');
        ylabel('PC1 projection (a.u.)'); xlabel('frame of the concatenated movie');
        title('PC1 time course of the corrected movie (slow ramp / steps = drift or state change)');
        for a = ax', xlim(a, [1 N]); box(a, 'off'); segLines(a, segBreaks); end
        linkaxes(ax, 'x');
        title(tl, [ttl ' - motion-correction time courses'], 'FontWeight', 'bold');
        saveFig(f, fullfile(outDir, '01_timecourses.png'));

        %% FIG 2: PC1 of the corrected movie
        nTop = M.opts.nTop;
        f = newfig([1300 800]);
        tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        lim = prctile1([M.topMean(:, :, 1); M.botMean(:, :, 1)], [1 99.7]);
        nexttile; imagesc(M.topMean(:, :, 1), lim); axis image off; colormap(gca, gray); cb = colorbar; ylabel(cb, 'mean F (a.u.)');
        title(sprintf('PC1 top frames: mean of the %d frames\nwith the highest PC1 projection', nTop));
        nexttile; imagesc(M.botMean(:, :, 1), lim); axis image off; colormap(gca, gray); cb = colorbar; ylabel(cb, 'mean F (a.u.)');
        title(sprintf('PC1 bottom frames: mean of the %d frames\nwith the lowest PC1 projection', nTop));
        nexttile; d = M.topMean(:, :, 1) - M.botMean(:, :, 1);
        imagesc(d, max(abs(d(:))) * [-1 1]); axis image off; colormap(gca, bwr()); cb = colorbar; ylabel(cb, '\DeltaF (a.u.)');
        title({'top - bottom', 'edge dipoles = motion; blobs = activity / z'});
        nexttile; plot(M.regDX, '.-', 'MarkerSize', 12); hold on; yline(0.5, 'k--');
        legend({'rigid', 'mean NR', 'max NR', 'mean(rigid+NR)', '0.5 px criterion'}, 'Location', 'northeast', 'Box', 'off');
        xlabel('PC #'); ylabel('offset between top and bottom means (px)'); box off
        title(sprintf('Registration offsets (Suite2p metric); max over PCs of mean(rigid+NR) = %.2f px', max(M.regDX(:, 4))));
        nexttile([1 2]); plot(M.frameIdx, M.tPC(:, 1), 'k'); hold on; segLines(gca, segBreaks);
        xlabel('frame of the concatenated movie'); ylabel('PC1 projection (a.u.)'); xlim([1 N]); box off
        title('PC1 time course (sampled frames)');
        title(tl, [ttl ' - PC1 of the motion-corrected movie'], 'FontWeight', 'bold');
        saveFig(f, fullfile(outDir, '02_PC1_corrected.png'));

        %% FIG 3: PC1 of every raw segment
        f = newfig([1500 330 * nSeg + 80]);
        tl = tiledlayout(nSeg, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
        for i = 1:nSeg
            lim = prctile1([Seg(i).topMean(:); Seg(i).botMean(:)], [1 99.7]);
            nexttile; imagesc(Seg(i).topMean, lim); axis image off; colormap(gca, gray);
            title({strrep(Seg(i).name, '_', '\_'), 'PC1 top frames mean (raw)'});
            nexttile; imagesc(Seg(i).botMean, lim); axis image off; colormap(gca, gray);
            title({sprintf('%d frames', Seg(i).T), 'PC1 bottom frames mean (raw)'});
            nexttile; d = Seg(i).topMean - Seg(i).botMean;
            imagesc(d, max(abs(d(:))) * [-1 1]); axis image off; colormap(gca, bwr()); cb = colorbar; ylabel(cb, '\DeltaF (a.u.)');
            title({sprintf('top - bottom, rigid offset %.2f px', Seg(i).regDX(1, 1)), 'raw: no motion correction'});
            nexttile; plot(Seg(i).frameIdx, Seg(i).pc1, 'k'); xlim([1 Seg(i).T]); box off
            xlabel('frame within the segment'); ylabel('PC1 projection (a.u.)'); title('PC1 time course (raw)');
        end
        title(tl, [ttl ' - PC1 of each raw segment (read directly from the .sbx)'], 'FontWeight', 'bold');
        saveFig(f, fullfile(outDir, '03_PC1_raw_segments.png'));

        %% FIG 4: mean image raw vs corrected
        f = newfig([1500 520]);
        tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        lim = prctile1([M.meanImgRaw(:); M.meanImg(:)], [1 99.7]);
        nexttile; imagesc(M.meanImgRaw, lim); axis image off; colormap(gca, gray); cb = colorbar; ylabel(cb, 'mean F (a.u.)');
        title({'RAW mean image', lbl.rawNote, sprintf('sharpness %.4f (normalised gradient)', shp.raw)});
        nexttile; imagesc(M.meanImg, lim); axis image off; colormap(gca, gray); cb = colorbar; ylabel(cb, 'mean F (a.u.)');
        title({'CORRECTED mean image', sprintf('sharpness %.4f (x%.2f of raw)', shp.cor, shp.cor / shp.raw)});
        nexttile; d = M.meanImg - M.meanImgRaw;
        imagesc(d, max(abs(d(:))) * [-1 1]); axis image off; colormap(gca, bwr()); cb = colorbar; ylabel(cb, '\DeltaF (a.u.)');
        title({'corrected - raw', 'where motion correction moved signal'});
        title(tl, [ttl ' - mean image before and after motion correction (same pixels)'], 'FontWeight', 'bold');
        saveFig(f, fullfile(outDir, '04_mean_image_raw_vs_corrected.png'));

        %% FIG 5: zoomed patches
        sm = imgaussfilt(M.meanImg, 15);
        [gx, gy] = imgradientxy(M.meanImg / mean(M.meanImg(:)), 'central');
        en = imgaussfilt(hypot(gx, gy), 15) .* (sm > median(sm(:)));
        ploc = zeros(2, 2);                                    % [row col] centre of each patch
        [~, i1] = max(sm(:));     [ploc(1, 1), ploc(1, 2)] = ind2sub(size(sm), i1);
        [~, i2] = max(en(:));     [ploc(2, 1), ploc(2, 2)] = ind2sub(size(en), i2);
        pname = {'brightest region', 'most structured region (strongest edges)'};
        zs = min(o.zoomSize, min(size(M.meanImg)));
        f = newfig([1100 1050]);
        tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        o4 = double(S.crop.out);
        for p = 1:2
            r0 = min(max(round(ploc(p, 1) - zs/2), 1), size(M.meanImg, 1) - zs + 1);
            c0 = min(max(round(ploc(p, 2) - zs/2), 1), size(M.meanImg, 2) - zs + 1);
            rr = r0:r0+zs-1; cc = c0:c0+zs-1;
            A = M.meanImgRaw(rr, cc); Bz = M.meanImg(rr, cc);
            lim = prctile1([A(:); Bz(:)], [1 99.7]);
            loc = sprintf('raw-frame rows %d-%d, cols %d-%d', o4(1)+r0-1, o4(1)+r0+zs-2, o4(2)+c0-1, o4(2)+c0+zs-2);
            nexttile; imagesc(A, lim); axis image off; colormap(gca, gray);
            title({sprintf('RAW, %s', pname{p}), sprintf('%s; sharpness %.4f', loc, sharpness(A))});
            nexttile; imagesc(Bz, lim); axis image off; colormap(gca, gray);
            title({sprintf('CORRECTED, %s', pname{p}), sprintf('same pixels; sharpness %.4f (x%.2f)', sharpness(Bz), sharpness(Bz) / sharpness(A))});
        end
        title(tl, {[ttl ' - zoomed mean image (zig-zag edges = scan-line offset; smear = residual motion)'], ...
            lbl.rawNote}, 'FontWeight', 'bold');
        saveFig(f, fullfile(outDir, '05_zoom_raw_vs_corrected.png'));

        %% FIG 6: bidirectional scan phase
        if isstruct(S.bidi)
            B = S.bidi;
            f = newfig([1400 480]);
            tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
            nexttile; hold on
            plot(B.lags, B.cc, 'o-', 'Color', cRaw, 'LineWidth', 1.2);
            leg = {sprintf('before correction (peak at %+.2f px)', B.lag)};
            if ~isempty(B.ccAfter)
                plot(B.lags, B.ccAfter, 's-', 'Color', 'k', 'LineWidth', 1.2);
                leg{end+1} = sprintf('after correction (peak at %+.2f px)', B.lagAfter);
            end
            xline(0, ':'); legend(leg, 'Location', 'south', 'Box', 'off');
            xlabel('lag of odd rows relative to even rows (px)'); ylabel('row-pair correlation'); box off
            title('Even/odd scan-line correlation vs lag');
            nexttile; hold on
            ctr = mean(B.bandCols, 2);
            bw = bar(ctr, [B.lagBands(:), B.lagBandsAfter(:)], 'grouped');
            bw(1).FaceColor = cRaw; bw(2).FaceColor = [0.2 0.2 0.2];
            legend({'before', 'after correction'}, 'Location', 'best', 'Box', 'off');
            xlabel('column of the raw frame (band centre, px)'); ylabel('even/odd offset (px)'); box off
            title('Offset per column band (differences = a single shift under-corrects)');
            nexttile; axis off
            txt = {sprintf('mode: %s', B.mode), ...
                sprintf('estimate (before): %+.2f px   [per-frame median %+.2f, MAD %.2f, n = %d frames]', ...
                    B.lag, B.lagMedian, B.lagMAD, B.nFrames), ...
                sprintf('applied: odd rows %+.2f px, even rows %+.2f px', B.applied(1), B.applied(2)), ...
                sprintf('residual (after): %+.2f px', B.lagAfter), ...
                sprintf('peak correlation %.2f, reliable = %d', B.peakCorr, B.reliable), '', ...
                'lag > 0: features of odd raw rows lie right of the', ...
                'same features in even rows; applied shift = [-lag/2, +lag/2].'};
            text(0, 1, txt, 'VerticalAlignment', 'top', 'FontSize', 11, 'Interpreter', 'none');
            title(tl, [ttl ' - bidirectional scan-phase offset'], 'FontWeight', 'bold');
            saveFig(f, fullfile(outDir, '06_bidi_scan_phase.png'));
        end

        %% FIG 7: early vs late
        f = newfig([1600 800]);
        tl = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
        rows = {'raw', 'RAW'; 'cor', 'CORRECTED'};
        lim = prctile1([el.raw.first(:); el.raw.last(:); el.cor.first(:); el.cor.last(:)], [1 99.7]);
        for r = 1:2
            E = el.(rows{r, 1});
            nexttile((r - 1) * 4 + 1); imagesc(E.first, lim); axis image off; colormap(gca, gray);
            title({sprintf('%s: first time bin', rows{r, 2}), sprintf('frames 1-%d', round(N / o.nBins))});
            nexttile((r - 1) * 4 + 2); imagesc(E.last, lim); axis image off; colormap(gca, gray);
            title({sprintf('%s: last time bin', rows{r, 2}), sprintf('frames %d-%d', N - round(N / o.nBins) + 1, N)});
            nexttile((r - 1) * 4 + 3); imagesc(E.diff, max(abs(E.diff(:))) * [-1 1]); axis image off; colormap(gca, bwr()); cb = colorbar;
            ylabel(cb, '\DeltaF (a.u.)');
            title({sprintf('%s: last - first', rows{r, 2}), ...
                sprintf('shift dy %+.2f dx %+.2f px; corr(diff, mean) %+.2f; corr(diff, gradient) %.2f', ...
                    E.dy, E.dx, E.corrGain, E.corrShift)}, 'FontSize', 9);
        end
        nexttile(4); hold on
        plot(1:o.nBins, el.raw.sharpPerBin, 'o-', 'Color', cRaw);
        plot(1:o.nBins, el.cor.sharpPerBin, 'o-', 'Color', 'k');
        legend({'raw', 'corrected'}, 'Location', 'best', 'Box', 'off');
        xlabel('time bin'); ylabel('sharpness (norm. gradient)'); box off
        title('Mean-image sharpness per time bin');
        nexttile(8); imagesc(binCorr, [min(binCorr(:)) 1]); axis image; colorbar; colormap(gca, parula);
        xlabel('time bin'); ylabel('time bin'); title('Corrected: bin-to-bin image correlation');
        title(tl, [ttl ' - first vs last time bin (corr(diff, mean) high = gain; corr(diff, gradient) high = shift)'], ...
            'FontWeight', 'bold');
        saveFig(f, fullfile(outDir, '07_early_vs_late.png'));

        %% FIG 8: kymographs
        K = M.kymo;
        f = newfig([1500 1000]);
        tl = tiledlayout(4, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
        nexttile(1, [4 1]); imagesc(M.meanImg); axis image off; colormap(gca, gray); hold on
        hh = (K.width - 1) / 2;
        [ly, lx] = size(M.meanImg);
        patch([0.5 lx+0.5 lx+0.5 0.5], K.row + [-hh -hh hh hh], 'c', 'FaceAlpha', 0.3, 'EdgeColor', 'c');
        patch(K.col + [-hh hh hh -hh], [0.5 0.5 ly+0.5 ly+0.5], 'm', 'FaceAlpha', 0.3, 'EdgeColor', 'm');
        title({'mean image', 'cyan = horizontal line, magenta = vertical line'});
        panels = {K.Hraw, K.tRaw, 'RAW, horizontal line (cyan)', 'x (px)'; ...
                  K.H, K.t, 'CORRECTED, horizontal line (cyan)', 'x (px)'; ...
                  K.Vraw, K.tRaw, 'RAW, vertical line (magenta)', 'y (px)'; ...
                  K.V, K.t, 'CORRECTED, vertical line (magenta)', 'y (px)'};
        for i = 1:4
            nexttile((i - 1) * 4 + 2, [1 3]);
            img = panels{i, 1}; t = panels{i, 2};
            imagesc(t([1 end]), [1 size(img, 1)], img, prctile1(img(:), [1 99])); colormap(gca, gray); box off
            ylabel(panels{i, 4}); title(sprintf('%s  (%d px band, %d-frame bins)', panels{i, 3}, K.width, K.binFrames));
            if i == 4, xlabel('frame of the concatenated movie'); end
            segLines(gca, segBreaks);
        end
        title(tl, [ttl ' - kymographs (straight stripes = stable; bending = motion; fading = z)'], 'FontWeight', 'bold');
        saveFig(f, fullfile(outDir, '08_kymographs.png'));

        %% summary
        perSeg = cell(nSeg, 1);
        for i = 1:nSeg
            rng = S.frameIndex(:, 1) == i;
            perSeg{i} = sprintf('  %-28s %6d frames  dy %+6.2f +- %.2f   dx %+6.2f +- %.2f   raw PC1 rigid offset %.2f px', ...
                segNames{i}, Seg(i).T, mean(S.dy(rng)), std(S.dy(rng)), mean(S.dx(rng)), std(S.dx(rng)), Seg(i).regDX(1, 1));
        end
        lines = [{sprintf('Motion-correction diagnostics: %s', o.name), ...
            sprintf('  %d frames, %d segment(s); output pixels %d x %d (raw-frame rect [%s])', N, nSeg, ...
                size(M.meanImg, 1), size(M.meanImg, 2), num2str(double(S.crop.out))), ...
            '', 'Rigid shifts per segment (mean +- SD, px):'}, perSeg(:)', {'', ...
            sprintf('Rigid shift SD: dy %.2f px, dx %.2f px; outlier frames %.2f %%', std(S.dy), std(S.dx), 100 * mean(S.bad)), ...
            sprintf('PC registration offset, max over PCs of mean(rigid+NR): %.3f px (criterion 0.5)', max(M.regDX(:, 4))), ...
            sprintf('Frame-to-%s correlation: raw median %.3f (1st pct %.3f); corrected median %.3f (1st pct %.3f)', ...
                lbl.ref, fcRaw(1), fcRaw(2), fcCor(1), fcCor(2)), ...
            sprintf('Raw definition: %s', lbl.rawNote), ...
            sprintf('Mean F, last / first time bin: raw %.3f, corrected %.3f', ...
                mean(M.meanFRaw(end-numel(b1)+1:end)) / F0raw, mean(M.meanF(end-numel(b1)+1:end)) / F0), ...
            sprintf('Mean-image sharpness: raw %.4f, corrected %.4f (x%.2f)', shp.raw, shp.cor, shp.cor / shp.raw), ...
            sprintf('First vs last bin (corrected): shift dy %+.2f dx %+.2f px; corr(diff, mean) %+.2f; corr(diff, gradient) %.2f', ...
                el.cor.dy, el.cor.dx, el.cor.corrGain, el.cor.corrShift)}];
        if isstruct(S.bidi)
            lines{end+1} = sprintf('Scan phase (even/odd rows): estimate %+.2f px, applied [%+.2f %+.2f], residual %+.2f px (mode %s)', ...
                S.bidi.lag, S.bidi.applied(1), S.bidi.applied(2), S.bidi.lagAfter, S.bidi.mode);
        else
            lines{end+1} = 'Scan phase: no record in the _shifts.mat (motion correction made without bidiShift)';
        end
        fid = fopen(fullfile(outDir, 'mc_diagnostics_summary.txt'), 'w');
        fprintf(fid, '%s\n', lines{:});
        fclose(fid);
        fprintf('%s\n', lines{:});

        D = struct('name', o.name, 'nFrames', N, 'segments', {segNames}, 'regDXmax', max(M.regDX(:, 4)), ...
            'dySD', std(S.dy), 'dxSD', std(S.dx), 'frameCorrRaw', fcRaw, 'frameCorrCorrected', fcCor, ...
            'sharpnessRaw', shp.raw, 'sharpnessCorrected', shp.cor, 'earlyLate', rmfield(el.cor, {'first', 'last', 'diff'}), ...
            'bidi', S.bidi);
        M.topMean = M.topMean(:, :, 1);                 % keep only PC1 images in the .mat
        M.botMean = M.botMean(:, :, 1);
        save(fullfile(outDir, 'mc_diagnostics.mat'), 'D', 'M', 'Seg', 'el', 'binCorr');
        end

        function S = segments(sbxFiles, outPng, varargin)
        % SEGMENTS  Mean and max projection of every .sbx segment that
        % will be concatenated into one motion-corrected movie, read directly from
        % the .sbx (no tif), plus how each segment sits relative to segment 1.
        %
        %   S = mc_diag_method.segments(sbxFiles, outPng)
        %   S = mc_diag_method.segments(sbxFiles, outPng, 'channel', 1, 'crop', [1 80 Inf Inf], ...
        %                           'nSample', 1500, 'first', 1, 'title', 'session name')
        %
        %   sbxFiles  cell of .sbx paths (extension optional), in concatenation order
        %   outPng    figure file; a .mat with the images is saved next to it
        %   channel   PMT channel (1 = green)
        %   crop      raw-frame crop, as sbx_motion_correct
        %   nSample   frames per segment, evenly spaced (mean and max are over these;
        %             the max of a subsample is a lower bound of the true max)
        %   first     frames dropped at the start of every file (1, as the pipeline)
        %
        % Figure (one column per segment):
        %   row 1  mean projection, row 2  max projection (same colour scale across
        %   segments), row 3  overlay of segment 1 (magenta) with this segment
        %   (green): white = aligned, colour fringes = the segment sits elsewhere.
        %   Titles give the shift (dy, dx in px) of each segment's mean image
        %   relative to segment 1 by phase correlation. These are RAW images, before
        %   motion correction: a step between segments is what the concatenated
        %   correction has to absorb.
        %
        % S: names, nFramesDisk, nSampled, shiftVs1 [nSeg x 2] (dy dx), peak.

        ip = inputParser;
        ip.addParameter('channel', 1);
        ip.addParameter('crop', [1 80 Inf Inf]);
        ip.addParameter('nSample', 1500);
        ip.addParameter('first', 1);
        ip.addParameter('title', '');
        ip.parse(varargin{:});
        o = ip.Results;

        sbxFiles = cellstr(sbxFiles);
        nSeg = numel(sbxFiles);
        meanImg = cell(1, nSeg);
        maxImg = cell(1, nSeg);
        names = cell(1, nSeg);
        nDisk = zeros(1, nSeg);
        nSamp = zeros(1, nSeg);
        for i = 1:nSeg
            P = sbx_method.projections(sbxFiles{i}, 'channel', o.channel, 'crop', o.crop, ...
                'nSample', o.nSample, 'first', o.first);
            meanImg{i} = P.mean;
            maxImg{i} = P.max;
            names{i} = P.name;
            nDisk(i) = P.nFramesDisk;
            nSamp(i) = P.nSampled;
        end

        % shift of every segment relative to segment 1
        shiftVs1 = zeros(nSeg, 2);
        peak = zeros(nSeg, 1);
        for i = 2:nSeg
            [shiftVs1(i, 1), shiftVs1(i, 2), peak(i)] = mc_diag_method.shift(meanImg{1}, meanImg{i});
        end

        % figure
        mlim = prctile1(cell2mat(cellfun(@(x) x(:), meanImg(:), 'UniformOutput', false)), [1 99.7]);
        xlim_ = prctile1(double(cell2mat(cellfun(@(x) x(:), maxImg(:), 'UniformOutput', false))), [1 99.7]);
        f = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 min(500 * nSeg + 150, 2400) 1000]);
        tl = tiledlayout(3, nSeg, 'TileSpacing', 'compact', 'Padding', 'compact');
        rowName = {'mean projection', 'max projection (sampled frames)', ...
            'overlay: segment 1 (magenta) + this segment (green)'};
        m1 = rescale01(meanImg{1}, mlim);
        for i = 1:nSeg
            nexttile(i);
            imagesc(meanImg{i}, mlim); axis image off; colormap(gca, gray);
            if i == 1
                sh = 'reference segment';
            else
                sh = sprintf('shift vs seg 1: dy %+.2f, dx %+.2f px', shiftVs1(i, 1), shiftVs1(i, 2));
            end
            title({strrep(names{i}, '_', '\_'), sprintf('%d frames on disk, %d sampled', nDisk(i), nSamp(i)), sh}, ...
                'FontSize', 9);
            if i == 1, rowLabel(rowName{1}); end
            nexttile(nSeg + i);
            imagesc(double(maxImg{i}), xlim_); axis image off; colormap(gca, gray);
            if i == 1, rowLabel(rowName{2}); end
            nexttile(2 * nSeg + i);
            if i == 1
                imagesc(m1); axis image off; colormap(gca, gray);
                title('(segment 1 alone)', 'FontSize', 9);
            else
                mk = rescale01(meanImg{i}, mlim);
                image(cat(3, m1, mk, m1)); axis image off;
            end
            if i == 1, rowLabel(rowName{3}); end
        end
        if isempty(o.title)
            [~, o.title] = fileparts(fileparts(sbxFiles{1}));
        end
        title(tl, sprintf('Raw segments to be concatenated: %s  (channel %d, before motion correction)', ...
            strrep(o.title, '_', '\_'), o.channel), 'FontWeight', 'bold');
        saveFig(f, outPng);

        S = struct('names', {names}, 'nFramesDisk', nDisk, 'nSampled', nSamp, ...
            'shiftVs1', shiftVs1, 'peak', peak);
        [d, n] = fileparts(outPng);
        save(fullfile(d, [n '.mat']), 'S', 'meanImg', 'maxImg');
        end

        function referenceFigure(refTif, mcPath, outPng, name)
        % REFERENCEFIGURE  Figure of the reference-to-session registration
        % (output of sbx_method.registerReference): the two images, their overlay, a
        % checkerboard, the reference before registration, the residual
        % misalignment between them, and the crop rectangles.
        %
        %   mc_diag_method.referenceFigure(refTif, mcPath, outPng, name)
        sessDir = fileparts(mcPath);
        R = load(fullfile(sessDir, 'all_registeration_imgs.mat'), 'registered', 'avg_mcvid', 'avg_refvid');
        green = double(R.avg_mcvid);
        red = double(R.registered);
        lim = @(x) interp1(linspace(0, 100, numel(x)), sort(x(:)), [1 99.7]);
        norm01 = @(x, l) min(max((x - l(1)) / max(l(2) - l(1), eps), 0), 1);
        g = norm01(green, lim(green));
        r = norm01(red, lim(red));

        % residual: phase correlation of gradient-magnitude images (two modalities)
        ed = @(x) imgradient(imgaussfilt(x, 1));
        [dy, dx, pk] = mc_diag_method.shift(ed(g), ed(r));
        ov = (red > 0) & (green > 0);
        ncc = corr(g(ov), r(ov));
        [yy, xx] = ndgrid(1:size(g, 1), 1:size(g, 2));
        tile = mod(floor((yy - 1) / 64) + floor((xx - 1) / 64), 2) == 0;
        cb = g .* tile + r .* ~tile;
        refCrop = sbx_method.cropOut(refTif);
        sessCrop = sbx_method.cropOut(mcPath);

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 1500 900]);
        tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        nexttile; imagesc(g); axis image off; colormap(gca, gray);
        title({'session mean image (functional channel)', 'motion-corrected movie'});
        nexttile; imagesc(r); axis image off; colormap(gca, gray);
        title({'reference (structural channel), registered', 'rigid, into the session frame'});
        nexttile; image(cat(3, r, g, zeros(size(g)))); axis image off;
        title({'overlay: session = green, reference = red', 'yellow = aligned; single colour = mismatch'});
        nexttile; imagesc(cb); axis image off; colormap(gca, gray);
        title({'checkerboard (64 px tiles): session / reference', 'structures should continue across tile borders'});
        nexttile; imagesc(double(R.avg_refvid)); axis image off; colormap(gca, gray);
        title({'reference mean image BEFORE registration', 'native crop of its own motion correction'});
        nexttile; axis off
        if isequal(refCrop, sessCrop), same = '   -> same raw pixels'; else, same = '   -> different pixels (offset used as start)'; end
        txt = {sprintf('residual shift (edge phase correlation): dy %+.2f, dx %+.2f px', dy, dx), ...
            sprintf('   correlation peak %.2f (low = unreliable)', pk), ...
            sprintf('intensity correlation in the overlap (NCC): %.2f', ncc), '', ...
            sprintf('reference crop.out [r1 c1 r2 c2]: %s', num2str(refCrop)), ...
            sprintf('session   crop.out [r1 c1 r2 c2]: %s', num2str(sessCrop)), same, '', ...
            sprintf('session image %d x %d px; reference %d x %d px', size(green), size(R.avg_refvid))};
        text(0, 1, txt, 'VerticalAlignment', 'top', 'FontSize', 11, 'Interpreter', 'none');
        title(tl, sprintf('%s - registration of the reference to the session (rigid)', strrep(name, '_', '\_')), ...
            'FontWeight', 'bold');
        if ~isfolder(fileparts(outPng)), mkdir(fileparts(outPng)); end
        try
            exportgraphics(f, outPng, 'Resolution', 150);
        catch
            saveas(f, outPng);
        end
        close(f);
        end

        function [dy, dx, peak] = shift(ref, img)
        % SHIFT  Rigid translation between two images by phase correlation
        % (Hann-tapered, sub-pixel by a parabola through the peak).
        %
        %   [dy, dx, peak] = mc_diag_method.shift(ref, img)
        %
        % (dy, dx) = shift of the content of img relative to ref, in px: img(y,x) ~
        % ref(y - dy, x - dx). peak = height of the correlation peak (0..1, higher =
        % more reliable). Same-size 2-D inputs; NaNs are treated as 0.

        ref = double(ref);
        img = double(img);
        ref(isnan(ref)) = 0;
        img(isnan(img)) = 0;
        [h, w] = size(ref);
        tap = (0.5 - 0.5 * cos(2*pi*(0:h-1)' / max(h-1, 1))) * (0.5 - 0.5 * cos(2*pi*(0:w-1) / max(w-1, 1)));
        a = (ref - mean(ref(:))) .* tap;
        b = (img - mean(img(:))) .* tap;
        R = fft2(b) .* conj(fft2(a));
        R = R ./ max(abs(R), eps);
        cc = fftshift(real(ifft2(R)));
        [peak, i] = max(cc(:));
        [py, px] = ind2sub(size(cc), i);
        dy = py - (floor(h/2) + 1);
        dx = px - (floor(w/2) + 1);
        if py > 1 && py < h
            d = cc(py-1, px) - 2*cc(py, px) + cc(py+1, px);
            if d < 0, dy = dy + 0.5 * (cc(py-1, px) - cc(py+1, px)) / d; end
        end
        if px > 1 && px < w
            d = cc(py, px-1) - 2*cc(py, px) + cc(py, px+1);
            if d < 0, dx = dx + 0.5 * (cc(py, px-1) - cc(py, px+1)) / d; end
        end
        end

    end
end


%% Local functions (used by the methods above)

function src = open_source(m, maxFrames, label)
% Movie source: an array, a tif path, or struct('path', ..., 'rows', ...,
% 'cols', ...). Fields: kind ('array' / 'tif'), data or path, rows, cols
% (pixels of the stored frame that are used), Ly, Lx, T, cls.
src = struct('kind', 'array', 'data', [], 'path', '', 'rows', [], 'cols', [], ...
             'Ly', 0, 'Lx', 0, 'T', 0, 'cls', '', 'sbx', []);
if isnumeric(m) || islogical(m)
    src.data = m;
    [H, W, n] = size(m);
    src.cls = class(m);
else
    if ischar(m) || isstring(m)
        m = struct('path', char(m));
    end
    if isstruct(m) && isfield(m, 'sbx')
        % Scanbox recording(s) read directly; several files are concatenated
        % in the order given, 'first' frames dropped from each (1 = as
        % sbx_motion_correct)
        src.kind = 'sbx';
        files = cellstr(m.sbx);
        s = struct('channel', 1, 'crop', [1 80 Inf Inf], 'first', 1, 'bidi', [0 0]);
        for f = fieldnames(s)'
            if isfield(m, f{1}) && ~isempty(m.(f{1}))
                s.(f{1}) = m.(f{1});
            end
        end
        info = cellfun(@sbx_method.info, files, 'UniformOutput', false);
        s.info = [info{:}];
        crop = double(s.crop(:)');
        s.rc = [max(crop(1:2), 1), min(crop(3:4), [s.info(1).rows s.info(1).cols])];
        nUse = max([s.info.nFramesDisk] - s.first, 0);
        s.cum = [0 cumsum(nUse)];
        src.sbx = s;
        if isfield(m, 'rows'), src.rows = m.rows(:)'; end
        if isfield(m, 'cols'), src.cols = m.cols(:)'; end
        H = s.rc(3) - s.rc(1) + 1;
        W = s.rc(4) - s.rc(2) + 1;
        n = s.cum(end);
        src.cls = 'uint16';
    else
        if ~isstruct(m) || ~isfield(m, 'path') || ~isfile(m.path)
            error('%s: give an array, an existing tif path, an sbx struct, or a struct with a ''path'' field.', label);
        end
        src.kind = 'tif';
        src.path = char(m.path);
        if isfield(m, 'rows'), src.rows = m.rows(:)'; end
        if isfield(m, 'cols'), src.cols = m.cols(:)'; end
        [H, W, n, src.cls] = tif_info(src.path, maxFrames);
    end
end
if isempty(src.rows), src.rows = 1:H; end
if isempty(src.cols), src.cols = 1:W; end
if max(src.rows) > H || max(src.cols) > W
    error('%s: rows/cols exceed the %dx%d frame.', label, H, W);
end
src.Ly = numel(src.rows);
src.Lx = numel(src.cols);
src.T = min(n, maxFrames);
end

function [H, W, n, cls] = tif_info(path, maxFrames)
% Frame size, class and number of pages (up to maxFrames) of a tif,
% walking the page headers without reading image data.
[t, cleanup] = open_tif(path); %#ok<ASGLU>
H = t.getTag('ImageLength');
W = t.getTag('ImageWidth');
cls = class(t.read());
n = 1;
while n < maxFrames && ~t.lastDirectory()
    t.nextDirectory();
    n = n + 1;
end
end

function [t, cleanup] = open_tif(path)
% Tiff object with libtiff warnings off; both restored by cleanup.
w = warning('off', 'imageio:tiffmexutils:libtiffWarning');
t = Tiff(path, 'r');
cleanup = onCleanup(@() close_tif(t, w));
end

function close_tif(t, w)
t.close();
warning(w);
end

function idx = sample_index(T, Ly, Lx, nSample)
% Frames for the PC metric: nSample if given, else 2000 if large frame or
% short movie, else 5000.
if nargin >= 4 && ~isempty(nSample)
    nSamp = min(nSample, T);
elseif max(Ly, Lx) > 700 || T < 5000
    nSamp = min(2000, T);
else
    nSamp = 5000;
end
idx = unique(round(linspace(1, T, nSamp)));
end

function F = sbx_src_read(src, gidx)
% Cropped frames (full stored frame, uint16) with global (concatenated)
% 1-based indices gidx, ascending, of an 'sbx' source.
s = src.sbx;
F = zeros(s.rc(3)-s.rc(1)+1, s.rc(4)-s.rc(2)+1, numel(gidx), 'uint16');
for f = 1:numel(s.info)
    sel = find(gidx > s.cum(f) & gidx <= s.cum(f+1));
    if isempty(sel)
        continue
    end
    f0 = s.first + gidx(sel) - s.cum(f) - 1;              % 0-based frame on disk
    F(:, :, sel) = sbx_method.read(s.info(f).path, f0, 'channel', s.channel, ...
        'crop', s.rc, 'bidi', s.bidi, 'info', s.info(f));
end
end

function X = read_frames(src, idx, yr, xr)
% Frames idx (ascending) cropped to yr, xr of the source pixels, in the
% source's class. A tif is walked page by page; only frames in idx are read.
R = src.rows(yr);
C = src.cols(xr);
if strcmp(src.kind, 'array')
    X = src.data(R, C, idx);
    return
end
if strcmp(src.kind, 'sbx')
    X = zeros(numel(R), numel(C), numel(idx), 'uint16');
    for k0 = 1:200:numel(idx)
        k = k0:min(k0 + 199, numel(idx));
        F = sbx_src_read(src, idx(k));
        X(:, :, k) = F(R, C, :);
    end
    return
end
X = zeros(numel(R), numel(C), numel(idx), src.cls);
[t, cleanup] = open_tif(src.path); %#ok<ASGLU>
cur = 1;
for k = 1:numel(idx)
    while cur < idx(k)
        t.nextDirectory();
        cur = cur + 1;
    end
    I = t.read();
    X(:,:,k) = I(R, C);
end
end

function mimg = mean_image(X)
% Mean over frames in double, a frame block at a time (no full double copy).
mimg = zeros(size(X, 1), size(X, 2));
for s = 1:500:size(X, 3)
    mimg = mimg + sum(X(:,:,s:min(s+499, end)), 3, 'double');
end
mimg = mimg / size(X, 3);
end

function [regDX, tPC, topMean, botMean] = pc_metric(X, o)
% Steps 2-5 on sampled frames X (ly x lx x nf). PCA via the nf x nf Gram
% matrix of the pixel-mean-subtracted frames, accumulated over pixel blocks
% (same PC time courses as an SVD of the nf x npix matrix).
[ly, lx, nf] = size(X);
nPC  = min(o.nPC, nf - 1);
nTop = min(o.nTop, floor(nf/2));
Xp = reshape(X, ly*lx, nf);
npix = ly*lx;
step = max(1, floor(2e8 / (8 * nf)));          % ~200 MB of doubles per block
G = zeros(nf);
for p0 = 1:step:npix
    pix = p0:min(p0 + step - 1, npix);
    B = double(Xp(pix, :));
    B = B - mean(B, 2);
    G = G + B' * B;
end
clear Xp B
G = (G + G') / 2;
[U, L] = eig(G, 'vector');
[L, order] = sort(L, 'descend');
U = U(:, order(1:nPC));
tPC = U .* sqrt(max(L(1:nPC), 0))';             % nf x nPC, = U*S of the SVD

% ---- 3-4. top/bottom means and their relative offsets
topMean = zeros(ly, lx, nPC, 'single');
botMean = zeros(ly, lx, nPC, 'single');
regDX   = zeros(nPC, 4);
maxShift = round(o.maxShiftFrac * min(ly, lx));
for k = 1:nPC
    [~, ord] = sort(tPC(:, k));
    botMean(:,:,k) = mean(X(:,:,ord(1:nTop)), 3);
    topMean(:,:,k) = mean(X(:,:,ord(end-nTop+1:end)), 3);

    ref = topMean(:,:,k);  img = botMean(:,:,k);
    % rigid
    [dy, dx] = phasecorr_shift(ref, img, maxShift, o.smoothSigma);
    rig = [dy dx];
    img = fourier_shift(img, dy, dx);         % remove rigid part
    % non-rigid on overlapping blocks
    nr = nonrigid_shifts(ref, img, o.blockSize, o.maxShiftNR, o.smoothSigma);
    nrLen  = sqrt(sum(nr.^2, 2));
    totLen = sqrt(sum((nr + rig).^2, 2));
    regDX(k,:) = [norm(rig), mean(nrLen), max(nrLen), mean(totLen)];
end
end

function [r, H, V, meanF, fr, binImg] = stream_pass(src, yr, xr, mimg, row, col, h, chunk, stride, nBins)
% One pass over frames fr = 1:stride:T, chunk at a time: Pearson correlation
% of each (cropped) frame with mimg, the band-averaged profile along the
% horizontal (H, lx x nS) and vertical (V, ly x nS) kymograph lines, the
% mean fluorescence of each frame, and (nBins > 0) the mean image of each of
% nBins consecutive time bins of the whole movie.
T = src.T;
fr = 1:stride:T;
nS = numel(fr);
R = src.rows(yr);
C = src.cols(xr);
[ly, lx] = deal(numel(R), numel(C));
mz = mimg(:) - mean(mimg(:));
mz = mz / norm(mz);
r = zeros(nS, 1);
meanF = zeros(nS, 1);
H = zeros(lx, nS, 'single');
V = zeros(ly, nS, 'single');
binSum = zeros(ly, lx, max(nBins, 1));
binCnt = zeros(1, max(nBins, 1));
binOf = min(max(nBins, 1), ceil(fr / T * max(nBins, 1)));
isTif = strcmp(src.kind, 'tif');
if isTif
    [t, cleanup] = open_tif(src.path); %#ok<ASGLU>
    cur = 1;
end
nextReport = 0.1;
for s = 1:chunk:nS
    k = s:min(s + chunk - 1, nS);
    f = fr(k);
    switch src.kind
        case 'tif'
            F = zeros(ly, lx, numel(f), src.cls);
            for i = 1:numel(f)
                while cur < f(i)
                    t.nextDirectory();
                    cur = cur + 1;
                end
                I = t.read();
                F(:, :, i) = I(R, C);
            end
        case 'sbx'
            F = sbx_src_read(src, f);
            F = F(R, C, :);
        otherwise
            F = src.data(R, C, f);
    end
    H(:, k) = reshape(mean(single(F(row-h:row+h, :, :)), 1), lx, []);
    V(:, k) = reshape(mean(single(F(:, col-h:col+h, :)), 2), ly, []);
    D = double(reshape(F, [], numel(f)));
    meanF(k) = mean(D, 1)';
    if nBins > 0
        for b = unique(binOf(k))
            sel = binOf(k) == b;
            binSum(:, :, b) = binSum(:, :, b) + sum(double(F(:, :, sel)), 3);
            binCnt(b) = binCnt(b) + nnz(sel);
        end
    end
    D = D - mean(D, 1);
    r(k) = (mz' * D) ./ sqrt(sum(D.^2, 1));
    if k(end) / nS >= nextReport
        fprintf('    %3.0f%%\n', 100 * k(end) / nS);
        nextReport = nextReport + 0.1;
    end
end
binImg = [];
if nBins > 0
    binImg = single(binSum ./ reshape(max(binCnt, 1), 1, 1, []));
end
end

function [H, V, t] = bin_profiles(H, V, nBin)
% Average line profiles (position x frame) over bins of nBin frames.
% t = centre frame of each bin.
T = size(H, 2);
nb = floor(T / nBin);
if nb < 1
    nBin = T;
    nb = 1;
end
H = reshape(mean(reshape(H(:, 1:nb*nBin), size(H, 1), nBin, nb), 2), size(H, 1), nb);
V = reshape(mean(reshape(V(:, 1:nb*nBin), size(V, 1), nBin, nb), 2), size(V, 1), nb);
t = ((1:nb) - 0.5) * nBin + 0.5;
end

function [row, col] = pick_kymo_lines(mimg, h, sig)
% Row and column whose band (+/- h px) crosses the most structure: highest
% variance of the high-pass filtered mean image. Avoids the outer edges,
% where the high-pass filter is unreliable.
k = exp(-(-ceil(3*sig):ceil(3*sig)).^2 / (2*sig^2));
k = k / sum(k);
blur = @(x) conv2(k, k, x, 'same');
hp = mimg - blur(mimg) ./ blur(ones(size(mimg)));   % normalized at the edges
[ly, lx] = size(mimg);
m = min([h + ceil(sig), floor(ly/4), floor(lx/4)]);
band = ones(2*h + 1, 1) / (2*h + 1);
rowScore = conv(var(hp(:, m+1:lx-m), 0, 2), band, 'same');
colScore = conv(var(hp(m+1:ly-m, :), 0, 1)', band, 'same');
rowScore([1:m, ly-m+1:ly]) = -Inf;
colScore([1:m, lx-m+1:lx]) = -Inf;
[~, row] = max(rowScore);
[~, col] = max(colScore);
end

function [dy, dx, cmax] = phasecorr_shift(ref, img, maxShift, sig)
% Phase correlation: whiten both spectra, multiply, smooth in Fourier
% domain with a Gaussian, find peak within +/- maxShift, subpixel refine.
[ly, lx] = size(ref);
tap = spatial_taper(ly, lx, 2*sig);
ref = clip_pct(ref);  img = clip_pct(img);
Fr = fft2((ref - mean(ref(:))) .* tap);
Fi = fft2((img - mean(img(:))) .* tap);
Fr = Fr ./ (abs(Fr) + 1e-5);
Fi = Fi ./ (abs(Fi) + 1e-5);
G  = gauss_fourier(ly, lx, sig);
cc = real(ifft2(Fr .* conj(Fi) .* G));
cc = fftshift(cc);
cy = floor(ly/2) + 1;  cx = floor(lx/2) + 1;
maxShift = max(1, min([maxShift, cy-2, cx-2]));
win = cc(cy-maxShift:cy+maxShift, cx-maxShift:cx+maxShift);
[cmax, idx] = max(win(:));
[iy, ix] = ind2sub(size(win), idx);
% 3-point parabolic subpixel refinement (Suite2p uses Kriging; same idea)
sy = subpix(win, iy, ix, 1);  sx = subpix(win, iy, ix, 2);
dy = iy - maxShift - 1 + sy;
dx = ix - maxShift - 1 + sx;
end

function s = subpix(w, iy, ix, dim)
s = 0;
if dim == 1
    if iy > 1 && iy < size(w,1), a = w(iy-1,ix); b = w(iy,ix); c = w(iy+1,ix); else, return; end
else
    if ix > 1 && ix < size(w,2), a = w(iy,ix-1); b = w(iy,ix); c = w(iy,ix+1); else, return; end
end
den = a - 2*b + c;
if den ~= 0, s = 0.5 * (a - c) / den; end
s = max(-0.5, min(0.5, s));
end

function nr = nonrigid_shifts(ref, img, bs, maxNR, sig)
% Overlapping (>=50%) square blocks; phase-correlate each block.
[ly, lx] = size(ref);
bs = min(bs, [ly lx]);
ny = max(1, ceil(1.5 * ly / bs(1)));  nx = max(1, ceil(1.5 * lx / bs(2)));
ys = round(linspace(1, ly - bs(1) + 1, ny));
xs = round(linspace(1, lx - bs(2) + 1, nx));
nr = zeros(numel(ys)*numel(xs), 2);  n = 0;
for iy = ys
    for ix = xs
        n = n + 1;
        r = ref(iy:iy+bs(1)-1, ix:ix+bs(2)-1);
        m = img(iy:iy+bs(1)-1, ix:ix+bs(2)-1);
        [nr(n,1), nr(n,2)] = phasecorr_shift(r, m, maxNR, sig);
    end
end
end

function out = fourier_shift(img, dy, dx)
[ly, lx] = size(img);
ky = ifftshift((0:ly-1) - floor(ly/2))' / ly;
kx = ifftshift((0:lx-1) - floor(lx/2)) / lx;
out = real(ifft2(fft2(img) .* exp(-2i*pi*(ky*dy + kx*dx))));
end

function G = gauss_fourier(ly, lx, sig)
ky = ifftshift((0:ly-1) - floor(ly/2))' / ly;
kx = ifftshift((0:lx-1) - floor(lx/2)) / lx;
G = exp(-2*pi^2*sig^2*(ky.^2 + kx.^2));
end

function tap = spatial_taper(ly, lx, slope)
my = 1 ./ (1 + exp(-(((ly-1)/2 - abs((0:ly-1)' - (ly-1)/2)) - 2*slope) / slope));
mx = 1 ./ (1 + exp(-(((lx-1)/2 - abs((0:lx-1)  - (lx-1)/2)) - 2*slope) / slope));
tap = my * mx;
end

function x = clip_pct(x)
v = sort(x(:));
lo = v(max(1, round(0.01*numel(v))));  hi = v(max(1, round(0.99*numel(v))));
x = min(max(x, lo), hi);
end

function plot_mc_diag_method_metrics(M)
d = M.topMean(:,:,1) - M.botMean(:,:,1);
f1 = figure('Name', 'PC1 top/bottom', 'Color', 'w', 'Position', [100 100 1200 380]);
subplot(1,3,1); imagesc(M.topMean(:,:,1)); axis image off; colormap(gca, gray); title('PC1 top-frames mean');
subplot(1,3,2); imagesc(M.botMean(:,:,1)); axis image off; colormap(gca, gray); title('PC1 bottom-frames mean');
subplot(1,3,3); imagesc(d, max(abs(d(:)))*[-1 1]); axis image off; colormap(gca, bwr_map); colorbar; title('top - bottom');

f2 = figure('Name', 'PC1 time course', 'Color', 'w', 'Position', [100 520 900 250]);
plot(M.frameIdx, M.tPC(:,1), 'k'); xlabel('frame'); ylabel('PC1 projection'); box off; title('PC1 across frames');

f3 = figure('Name', 'Registration offsets', 'Color', 'w', 'Position', [1000 520 600 400]);
plot(M.regDX, '.-', 'MarkerSize', 12); box off;
legend({'rigid', 'mean NR', 'max NR', 'mean(rigid+NR)'}, 'Location', 'northeast'); legend boxoff;
xlabel('PC #'); ylabel('offset length (px)'); title(sprintf('max over PCs: %.3f px', max(M.regDX(:,4))));
drawnow;
M.figs = [f1 f2 f3];
end

function plot_frame_corr(M)
% Frame-to-mean correlation over time and its distribution (raw vs
% corrected if the raw movie was given).
hasRaw = ~isempty(M.frameCorrRaw);
figure('Name', 'Frame-to-mean correlation', 'Color', 'w', 'Position', [100 100 1200 330]);
subplot(1,4,1:3); hold on
if hasRaw
    plot(M.streamIdxRaw, M.frameCorrRaw, 'Color', [0.85 0.33 0.10]);
end
plot(M.streamIdx, M.frameCorr, 'k');
xlabel('frame'); ylabel('corr. with mean image'); box off
ok = ~isnan(M.frameCorr);    % NaN = flat frame
title(sprintf('median %.3f, 1st pct %.3f (corrected)', median(M.frameCorr(ok)), prctile1(M.frameCorr(ok))));
subplot(1,4,4); hold on
edges = linspace(min([M.frameCorr; M.frameCorrRaw; 0.99]), 1, 60);
if hasRaw
    histogram(M.frameCorrRaw, edges, 'FaceColor', [0.85 0.33 0.10], 'EdgeColor', 'none');
end
histogram(M.frameCorr, edges, 'FaceColor', 'k', 'EdgeColor', 'none');
xlabel('corr. with mean image'); ylabel('frames'); box off
if hasRaw
    legend({'raw', 'corrected'}, 'Location', 'northwest'); legend boxoff
end
drawnow;
end

function plot_kymographs(M)
% Mean image with the two lines, and the kymograph along each (raw above
% corrected if the raw movie was given). Time runs left to right.
K = M.kymo;
hasRaw = ~isempty(K.Hraw);
nr = 2 + 2*hasRaw;
figure('Name', 'Kymographs', 'Color', 'w', 'Position', [100 100 1400 220*nr]);
subplot(nr, 4, 1 + 4*(0:nr-1));
imagesc(M.meanImg); axis image off; colormap(gca, gray); hold on
h = (K.width - 1) / 2;
[ly, lx] = size(M.meanImg);
patch([0.5 lx+0.5 lx+0.5 0.5], K.row + [-h -h h h] + [-0.5 -0.5 0.5 0.5], 'c', ...
    'FaceAlpha', 0.3, 'EdgeColor', 'c');
patch(K.col + [-h h h -h] + [-0.5 0.5 0.5 -0.5], [0.5 0.5 ly+0.5 ly+0.5], 'm', ...
    'FaceAlpha', 0.3, 'EdgeColor', 'm');
title({'mean image', 'cyan = horizontal line, magenta = vertical'});
panels = {K.H, K.t, 'horizontal line (cyan)', 'x (px)'; ...
          K.V, K.t, 'vertical line (magenta)', 'y (px)'};
if hasRaw
    panels = [{K.Hraw, K.tRaw, 'RAW horizontal line (cyan)', 'x (px)'}; panels(1,:); ...
              {K.Vraw, K.tRaw, 'RAW vertical line (magenta)', 'y (px)'}; panels(2,:)];
    panels(2:2:4, 3) = strcat({'CORRECTED '}, panels(2:2:4, 3));
end
for i = 1:nr
    [img, t, ttl, yl] = panels{i, :};
    subplot(nr, 4, (2:4) + 4*(i-1));
    imagesc(t([1 end]), [1 size(img, 1)], img, prctile1(img(:), [1 99]));
    colormap(gca, gray); box off
    ylabel(yl); title(sprintf('%s  (%d px band, %d-frame bins)', ttl, K.width, K.bin));
    if i == nr, xlabel('frame'); end
end
drawnow;
end

function c = bwr_map
n = 128; a = linspace(0,1,n)';
c = [[a a ones(n,1)]; [ones(n,1) flipud(a) flipud(a)]];
end

function [M, Seg, S, segNames, lbl] = fromFiles(mcPath, sbxFiles, o)
% Metrics by reading the corrected tif and the raw .sbx files.
sbxFiles = cellstr(sbxFiles);
nSeg = numel(sbxFiles);
[mcDir, mcName] = fileparts(mcPath);
shiftsPath = fullfile(mcDir, [mcName '_shifts.mat']);
need = {'dy', 'dx', 'cmax', 'bad', 'nonrigid', 'frameIndex', 'crop', 'outRows', 'outCols', 'params', 'bidi'};
have = who('-file', shiftsPath);
S = load(shiftsPath, need{ismember(need, have)});
for f = need
    if ~isfield(S, f{1}), S.(f{1}) = []; end
end
if isempty(S.crop) || isempty(S.outRows)
    error('%s has no crop / outRows; rerun the motion correction.', shiftsPath);
end
first = 1;
if isstruct(S.params) && isfield(S.params, 'dropFirstFrame')
    first = double(S.params.dropFirstFrame);
end
mk = @(files) struct('sbx', {files}, 'channel', o.channel, 'crop', double(S.crop.raw(:)'), ...
    'first', first, 'bidi', [0 0], 'rows', S.outRows, 'cols', S.outCols);
segNames = cellfun(@(f) regexprep(f, '.*[\\/]|\.sbx$', ''), sbxFiles, 'UniformOutput', false);

fprintf('mc_diagnostics: %s\n  corrected movie ...\n', o.name);
M = mc_diag_method.metrics(mcPath, 'rawMov', mk(sbxFiles), 'rawPC', false, 'nPC', o.nPC, ...
    'nSample', o.nSample, 'streamStride', o.stride, 'nBins', o.nBins, 'plot', false);
M.kymo.binFrames = M.kymo.bin * o.stride;

% PC metric on every raw segment (read from the .sbx)
Seg = repmat(struct('name', '', 'T', 0, 'frameIdx', [], 'pc1', [], 'topMean', [], 'botMean', [], ...
    'regDX', [], 'meanF', [], 'streamIdx', [], 'frameCorr', []), 1, nSeg);
for i = 1:nSeg
    fprintf('  raw segment %d/%d (%s) ...\n', i, nSeg, segNames{i});
    Mi = mc_diag_method.metrics(mk(sbxFiles(i)), 'nPC', o.segNPC, 'nSample', min(o.nSample, 1500), ...
        'streamStride', o.stride, 'nBins', 0, 'plot', false);
    Seg(i) = struct('name', segNames{i}, 'T', Mi.movSize(3), 'frameIdx', Mi.frameIdx, ...
        'pc1', Mi.tPC(:, 1), 'topMean', Mi.topMean(:, :, 1), 'botMean', Mi.botMean(:, :, 1), ...
        'regDX', Mi.regDX, 'meanF', Mi.meanF, 'streamIdx', Mi.streamIdx, 'frameCorr', Mi.frameCorr);
end
lbl = struct('rawNote', 'raw = no motion or scan-phase correction', 'ref', 'mean image');
end

function [M, Seg, S, segNames, lbl] = fromData(d, o)
% Metrics from variables kept in memory by sbx_motion_correct: per-frame
% statistics are taken as given, PCA / PC1 maps are computed on the sampled
% frames (the same mc_diag_method.metrics code as the path mode).
S = d.shifts;
for f = {'dy', 'dx', 'cmax', 'bad', 'nonrigid', 'frameIndex', 'crop', 'outRows', 'outCols', 'params', 'bidi'}
    if ~isfield(S, f{1}), S.(f{1}) = []; end
end
ns = size(d.corSample, 3);
N = numel(d.meanF.cor);
sIdx = double(d.sampleIdx(:));
fprintf('mc_diagnostics: %s (from variables, %d sampled frames of %d)\n', o.name, ns, N);
M = mc_diag_method.metrics(d.corSample, 'rawMov', d.rawSample, 'rawPC', false, 'nPC', o.nPC, ...
    'nSample', ns, 'streamStride', 1, 'nBins', 0, 'plot', false);
M.frameIdx = sIdx(M.frameIdx);                       % array positions -> frame numbers
M.streamIdx = (1:N)';
M.streamIdxRaw = (1:N)';
M.meanF = d.meanF.cor(:);
M.meanFRaw = d.meanF.raw(:);
M.frameCorr = d.frameCorr.cor(:);
M.frameCorrRaw = d.frameCorr.raw(:);
M.binImg = d.binImg.cor;
M.binImgRaw = d.binImg.raw;
M.movSize = [M.movSize(1:2) N];
M.kymo.t = interp1(1:ns, sIdx, M.kymo.t, 'linear', 'extrap');      % bin centres in frames
M.kymo.tRaw = interp1(1:ns, sIdx, M.kymo.tRaw, 'linear', 'extrap');
M.kymo.binFrames = round(M.kymo.bin * N / ns);

% PC1 of every raw segment, from the sampled raw frames of that segment
segNames = cellstr(d.segNames);
nSeg = numel(segNames);
first = 1;
if isstruct(S.params) && isfield(S.params, 'dropFirstFrame')
    first = double(S.params.dropFirstFrame);
end
fi = S.frameIndex;
file = fi(sIdx, 1);
within = fi(sIdx, 2) - first + 1;                    % frame number within the segment
Seg = repmat(struct('name', '', 'T', 0, 'frameIdx', [], 'pc1', [], 'topMean', [], 'botMean', [], ...
    'regDX', [], 'meanF', [], 'streamIdx', [], 'frameCorr', []), 1, nSeg);
for i = 1:nSeg
    fprintf('  raw segment %d/%d (%s) ...\n', i, nSeg, segNames{i});
    sel = find(file == i);
    inSeg = find(fi(:, 1) == i);
    Mi = mc_diag_method.metrics(d.rawSample(:, :, sel), 'nPC', o.segNPC, 'nSample', numel(sel), ...
        'streamStride', 1, 'nBins', 0, 'plot', false);
    Seg(i) = struct('name', segNames{i}, 'T', numel(inSeg), 'frameIdx', within(sel(Mi.frameIdx)), ...
        'pc1', Mi.tPC(:, 1), 'topMean', Mi.topMean(:, :, 1), 'botMean', Mi.botMean(:, :, 1), ...
        'regDX', Mi.regDX, 'meanF', d.meanF.raw(inSeg), 'streamIdx', (1:numel(inSeg))', ...
        'frameCorr', d.frameCorr.raw(inSeg));
end
note = 'raw = scan-phase corrected, no motion correction';
if isfield(d, 'rawNote') && ~isempty(d.rawNote), note = d.rawNote; end
ref = 'template';
if isfield(d, 'corrRef') && ~isempty(d.corrRef), ref = d.corrRef; end
lbl = struct('rawNote', note, 'ref', ref);
end

function f = newfig(sz)
f = figure('Visible', 'off', 'Color', 'w', 'Position', [50 50 sz]);
end

function saveFig(f, path)
try
    exportgraphics(f, path, 'Resolution', 150);
catch
    saveas(f, path);
end
close(f);
end

function segLines(ax, breaks)
for b = reshape(breaks, 1, [])
    xline(ax, b - 0.5, '--', 'Color', [0.5 0.5 0.5]);
end
end

function s = sharpness(img)
% mean gradient magnitude of the mean-normalised image (compare between
% images of the same pixels only)
img = double(img) / mean(double(img(:)));
[gx, gy] = imgradientxy(img, 'central');
s = mean(sqrt(gx(:).^2 + gy(:).^2));
end

function c = bwr()
n = 128; a = linspace(0, 1, n)';
c = [[a a ones(n, 1)]; [ones(n, 1) flipud(a) flipud(a)]];
end

function rowLabel(txt)
% vertical label at the left of a tile (its axis is off)
text(-0.04, 0.5, txt, 'Units', 'normalized', 'Rotation', 90, 'HorizontalAlignment', 'center', ...
    'FontSize', 9, 'FontWeight', 'bold');
end

function x = rescale01(img, lim)
x = min(max((double(img) - lim(1)) / max(lim(2) - lim(1), eps), 0), 1);
end

function v = prctile1(x, q)
% Percentile(s) without the Statistics Toolbox (default 1st); NaNs ignored.
if nargin < 2, q = 1; end
x = sort(double(x(:)));
x = x(~isnan(x));
if isscalar(x)
    v = x * ones(size(q));
else
    v = interp1(linspace(0, 100, numel(x)), x, q);
end
if numel(v) == 2 && v(1) == v(2), v = v(1) + [-1 1]; end   % flat image: valid clim
end
