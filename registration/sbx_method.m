classdef sbx_method
    % SBX_METHOD  Static helpers for Scanbox .sbx recordings and the files
    % built from them. Call as sbx_method.<method>(...); no object is needed.
    % (Motion correction itself is sbx_motion_correct.m; figures are in
    % mc_diag_method.)
    %
    % --- Read the .sbx ---
    %   s = sbx_method.info(sbx)                        geometry, channels, frame count
    %   Y = sbx_method.read(sbx, frames0, 'channel', 1, 'crop', ..., 'bidi', [0 0])
    %   outFiles = sbx_method.toTif(sbx, ...)           one tif per channel
    %   P = sbx_method.projections(sbx, ...)            mean / max over sampled frames
    %   B = sbx_method.bidiOffset(files, ...)           even/odd scan-line offset
    % --- Organise a day of recordings ---
    %   recs = sbx_method.findRecordings(dateDir, cfg)  sessions + references + behavior files
    %   movies = sbx_method.splitMovies(behavFile, ttlField)
    %   [iM, iS] = sbx_method.matchMovies(nBeh, nSbx)   pair behavior movies with segments
    %   cb = sbx_method.concatBehavior(movies, nUsed, match)
    % --- Motion-correction outputs ---
    %   r = sbx_method.cropOut(tifPath)                 output rectangle from the _shifts.mat
    %   sbx_method.registerReference(refTif, mcPath, regrefPath)

    methods (Static)

        function s = info(fname)
        % INFO  Frame geometry, channel layout and frame count of a
        % Scanbox .sbx file, read from its info .mat (follows sbxread and
        % sbx2tif_lily2025).
        %
        %   s = sbx_method.info('D:\data\250314_001_000')      % .sbx extension optional
        %
        % OUTPUT fields
        %   path        - full path to the .sbx file
        %   rows, cols  - frame size in px
        %   nchan       - number of channels interleaved in the file
        %   chs         - PMT channel numbers present (1 = green, 2 = red), in the
        %                 order they are interleaved
        %   frameBytes  - bytes per frame (all channels)
        %   nFramesDisk - number of complete frames in the file

        fname = char(fname);
        if endsWith(fname, '.sbx', 'IgnoreCase', true)
            fname = fname(1:end-4);
        end
        if ~isfile([fname '.sbx'])
            error('Cannot find %s.sbx', fname);
        end
        if ~isfile([fname '.mat'])
            error('Cannot find Scanbox info file %s.mat', fname);
        end
        S = load([fname '.mat'], 'info');
        info = S.info;
        if ~isfield(info, 'sz')
            info.sz = [512 796];        % sbxread default for old files
        end
        if ~isfield(info, 'scanmode')
            info.scanmode = 1;          % unidirectional
        end
        rows = info.recordsPerBuffer;
        if info.scanmode == 0
            rows = rows * 2;            % bidirectional
        end
        cols = info.sz(2);
        if info.sz(1) ~= rows
            warning('%s: info.sz(1)=%d but recordsPerBuffer gives %d rows; using %d.', ...
                fname, info.sz(1), rows, rows);
        end
        if isfield(info, 'chan')                 % Pizzabox / newer Scanbox
            nchan = info.chan.nchan;
            if isfield(info.chan, 'save')
                chs = find(info.chan.save);
            else
                chs = 1:nchan;
            end
        else
            switch info.channels
                case 1, nchan = 2; chs = [1 2];  % both PMTs
                case 2, nchan = 1; chs = 1;      % PMT 0
                case 3, nchan = 1; chs = 2;      % PMT 1
                otherwise, error('%s: unknown info.channels = %d', fname, info.channels);
            end
        end
        if numel(chs) ~= nchan
            error('%s: channel list %s does not match nchan = %d', fname, mat2str(chs), nchan);
        end
        d = dir([fname '.sbx']);
        s.path = [fname '.sbx'];
        s.rows = rows;
        s.cols = cols;
        s.nchan = nchan;
        s.chs = chs(:)';
        s.frameBytes = rows * cols * nchan * 2;
        s.nFramesDisk = floor(d.bytes / s.frameBytes);
        end

        function Y = read(sbxPath, frames, varargin)
        % READ  Read frames of one PMT channel straight from a Scanbox
        % .sbx file (no tif conversion).
        %
        %   Y = sbx_method.read(sbxPath, frames)
        %   Y = sbx_method.read(sbxPath, frames, 'channel', 1, 'crop', [1 80 Inf Inf], ...
        %                       'bidi', [0 0], 'info', [])
        %
        %   sbxPath  .sbx file (extension optional)
        %   frames   0-based frame numbers on disk (vector, any order, repeats ok)
        %   channel  PMT channel (1 = green, 2 = red)
        %   crop     [row1 col1 row2 col2] raw-frame crop (Inf = frame edge), the
        %            same convention as sbx_motion_correct
        %   bidi     [shiftOdd shiftEven] px: column shift of the odd / even raw rows
        %            applied before cropping (sbx_motion_correct's out.bidi.applied);
        %            [0 0] = raw frames
        %   info     sbx_method.info(sbxPath) if already available (saves a .mat load)
        %
        %   Y        uint16 [rows cols numel(frames)], in the order of 'frames'
        %            (same value convention as sbxread: the stored data is inverted).

        % plain name-value parsing (no inputParser: this also runs on thread workers)
        o = struct('channel', 1, 'crop', [1 80 Inf Inf], 'bidi', [0 0], 'info', []);
        for k = 1:2:numel(varargin)
            if ~isfield(o, varargin{k})
                error('sbx_method.read: unknown option ''%s''.', varargin{k});
            end
            o.(varargin{k}) = varargin{k+1};
        end
        info = o.info;
        if isempty(info)
            info = sbx_method.info(sbxPath);
        end
        crop = double(o.crop(:)');
        rc = [max(crop(1:2), 1), min(crop(3:4), [info.rows info.cols])];
        ci = find(info.chs == o.channel);
        if isempty(ci)
            error('%s: channel %d not recorded (channels: %s)', info.path, o.channel, mat2str(info.chs));
        end
        frames = double(frames(:)');
        if any(frames < 0 | frames >= info.nFramesDisk | frames ~= round(frames))
            error('%s: frame numbers must be integers in [0, %d].', info.path, info.nFramesDisk - 1);
        end
        [fs, ord] = sort(frames);
        n = numel(fs);
        Y = zeros(rc(3)-rc(1)+1, rc(4)-rc(2)+1, n, 'uint16');
        if n == 0
            return
        end
        fid = fopen(info.path, 'r');
        closeFile = onCleanup(@() fclose(fid));
        maxRun = 200;                                     % frames per fread
        brk = [0, find(diff(fs) ~= 1), n];                % contiguous runs
        pos = 0;
        for r = 1:numel(brk)-1
            for s0 = brk(r):maxRun:brk(r+1)-1
                m = min(maxRun, brk(r+1) - s0);
                fseek(fid, fs(s0+1) * info.frameBytes, 'bof');
                x = fread(fid, info.frameBytes / 2 * m, 'uint16=>uint16');
                if numel(x) ~= info.frameBytes / 2 * m
                    error('Short read in %s at frame %d', info.path, fs(s0+1));
                end
                % on disk: [chan col row frame], values inverted (as in sbxread)
                x = permute(reshape(x, info.nchan, info.cols, info.rows, m), [3 2 4 1]);
                if any(o.bidi ~= 0)
                    c1 = max(rc(2) - 3, 1);
                    c2 = min(rc(4) + 3, info.cols);
                    z = single(intmax('uint16') - x(:, c1:c2, :, ci));
                    z(1:2:end, :, :) = shiftCols(z(1:2:end, :, :), o.bidi(1));
                    z(2:2:end, :, :) = shiftCols(z(2:2:end, :, :), o.bidi(2));
                    Y(:, :, pos+(1:m)) = uint16(z(rc(1):rc(3), rc(2)-c1+1:rc(4)-c1+1, :));
                else
                    Y(:, :, pos+(1:m)) = intmax('uint16') - x(rc(1):rc(3), rc(2):rc(4), :, ci);
                end
                pos = pos + m;
            end
        end
        Y(:, :, ord) = Y;                                 % back to the requested order
        end

        function outFiles = toTif(sbxFile, varargin)
        % SBX2TIF  Convert a Scanbox .sbx recording to uint16 BigTIFF, one file
        % per channel (<name>_ch1.tif, <name>_ch2.tif)
        %
        %   outFiles = sbx2tif('D:\data\250314_001_000')          % .sbx optional
        %   outFiles = sbx2tif(f, 'channels', 1, 'crop', [1 80 Inf Inf])
        %
        % Reads the .sbx in chunks (fast, low memory; no frame-by-frame growing
        % arrays), inverts the values as sbxread does, and writes uncompressed
        % BigTIFF so files > 4 GB work.
        %
        % OPTIONAL NAME-VALUE INPUTS (defaults in brackets)
        %   'channels'   ([] = all recorded) PMT channels to write (1 = green, 2 = red)
        %   'crop'       ([1 80 Inf Inf]) [row1 col1 row2 col2] of the raw frame.
        %                Default drops the first 79 columns (bidirectional scan
        %                turnaround band), same as sbx_motion_correct and the old
        %                pipeline; [1 1 Inf Inf] keeps the full raw frame. Saved,
        %                with the raw frame size, to <name>_tifinfo.mat.
        %   'frames'     ([1 Inf]) [first last] frames to write, 1-based
        %   'outDir'     ([] = next to the .sbx) folder for the tifs
        %   'chunkSize'  (500) frames read per chunk
        %   'overwrite'  (false) replace existing tifs
        %   'verbose'    (true) print progress
        %
        % OUTPUT
        %   outFiles - cell array of the tif paths written (or already present)

        ip = inputParser;
        ip.addParameter('channels', []);
        ip.addParameter('crop', [1 80 Inf Inf]);
        ip.addParameter('frames', [1 Inf]);
        ip.addParameter('outDir', []);
        ip.addParameter('chunkSize', 500);
        ip.addParameter('overwrite', false);
        ip.addParameter('verbose', true);
        ip.parse(varargin{:});
        p = ip.Results;

        base = regexprep(char(sbxFile), '\.sbx$', '', 'ignorecase');
        s = sbx_method.info(base);
        chans = p.channels;
        if isempty(chans)
            chans = s.chs;
        end
        if ~all(ismember(chans, s.chs))
            error('%s: channel(s) %s not recorded (channels: %s)', base, mat2str(chans), mat2str(s.chs));
        end

        % crop and frame range, clipped to the file
        c = p.crop(:)';
        rc = [max(c(1:2), 1), min(c(3:4), [s.rows s.cols])];
        fr = [max(p.frames(1), 1), min(p.frames(2), s.nFramesDisk)];
        if fr(2) < fr(1)
            error('%s: no frames in range %s (file has %d)', base, mat2str(p.frames), s.nFramesDisk);
        end
        h = rc(3) - rc(1) + 1;
        w = rc(4) - rc(2) + 1;

        % output files
        [d, n] = fileparts(base);
        if ~isempty(p.outDir)
            d = p.outDir;
            if ~isfolder(d)
                mkdir(d);
            end
        end
        outFiles = arrayfun(@(ch) fullfile(d, sprintf('%s_ch%d.tif', n, ch)), chans, 'UniformOutput', false);
        todo = p.overwrite | ~isfile(outFiles);
        if ~any(todo)
            if p.verbose, fprintf('sbx2tif: %s already converted, skipping\n', n); end
            return
        end
        chans = chans(todo);
        todoIdx = find(todo);
        tw = cell(1, numel(chans));
        for k = 1:numel(chans)
            tw{k} = Tiff(outFiles{todoIdx(k)}, 'w8');   % BigTIFF
        end
        closeAll = onCleanup(@() cellfun(@close, tw));
        tags = struct('ImageLength', h, 'ImageWidth', w, ...
            'Photometric', Tiff.Photometric.MinIsBlack, 'BitsPerSample', 16, ...
            'SamplesPerPixel', 1, 'SampleFormat', Tiff.SampleFormat.UInt, ...
            'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky, ...
            'Compression', Tiff.Compression.None, 'RowsPerStrip', h);

        % read in chunks, write every channel
        fid = fopen(s.path, 'r');
        closeFile = onCleanup(@() fclose(fid));
        fseek(fid, (fr(1) - 1) * s.frameBytes, 'bof');
        nTotal = fr(2) - fr(1) + 1;
        ci = arrayfun(@(ch) find(s.chs == ch), chans);   % position of each channel on disk
        written = 0;
        tStart = tic;
        while written < nTotal
            m = min(p.chunkSize, nTotal - written);
            x = fread(fid, s.frameBytes / 2 * m, 'uint16=>uint16');
            if numel(x) ~= s.frameBytes / 2 * m
                error('%s: short read at frame %d', s.path, fr(1) + written);
            end
            % on disk: [chan col row frame], values inverted (as in sbxread)
            x = permute(reshape(x, s.nchan, s.cols, s.rows, m), [3 2 4 1]);   % [row col frame chan]
            x = intmax('uint16') - x(rc(1):rc(3), rc(2):rc(4), :, ci);
            for k = 1:numel(chans)
                for f = 1:m
                    if written + f > 1
                        tw{k}.writeDirectory();
                    end
                    tw{k}.setTag(tags);
                    tw{k}.write(x(:, :, f, k));
                end
            end
            written = written + m;
            if p.verbose
                fprintf('sbx2tif %s: %d/%d frames (%.0f s)\n', n, written, nTotal, toc(tStart));
            end
        end
        clear closeAll closeFile
        crop = struct('rawSize', [s.rows s.cols], 'raw', rc, 'outSize', [h w]); 
        frames = fr; 
        save(fullfile(d, [n '_tifinfo.mat']), 'crop', 'frames', 'outFiles');
        end

        function P = projections(sbx, varargin)
            % PROJECTIONS  Mean and max projection of one .sbx over evenly spaced frames.
            %   P = sbx_method.projections(sbx, 'channel', 1, 'crop', [1 80 Inf Inf], ...
            %                              'nSample', 1500, 'first', 1)
            %   P.mean (single), P.max (uint16; max of the sampled frames, a lower
            %   bound of the true max), P.nSampled, P.nFramesDisk, P.name
            o = struct('channel', 1, 'crop', [1 80 Inf Inf], 'nSample', 1500, 'first', 1);
            for k = 1:2:numel(varargin)
                o.(varargin{k}) = varargin{k+1};
            end
            info = sbx_method.info(sbx);
            avail = o.first:info.nFramesDisk - 1;
            idx = unique(round(linspace(avail(1), avail(end), min(o.nSample, numel(avail)))));
            sumImg = 0;
            mx = [];
            for k0 = 1:200:numel(idx)
                k = k0:min(k0 + 199, numel(idx));
                Y = sbx_method.read(info.path, idx(k), 'channel', o.channel, 'crop', o.crop, 'info', info);
                sumImg = sumImg + sum(double(Y), 3);
                if isempty(mx), mx = max(Y, [], 3); else, mx = max(mx, max(Y, [], 3)); end
            end
            [~, nm] = fileparts(info.path);
            P = struct('mean', single(sumImg / numel(idx)), 'max', mx, 'nSampled', numel(idx), ...
                'nFramesDisk', info.nFramesDisk, 'name', nm);
        end

        function B = bidiOffset(files, varargin)
            % BIDIOFFSET  Even/odd scan-line offset of (concatenated) .sbx recordings,
            % estimated on the raw frames: each odd raw row is cross-correlated with the
            % even row below it, summed over rows, averaged over frames. lag > 0 means
            % features of the odd rows lie lag columns right of those of the even rows.
            %   B = sbx_method.bidiOffset(files, 'channel', 1, 'crop', [1 80 Inf Inf], ...
            %         'first', 1, 'nFrames', 200, 'maxLag', 8, 'bands', 4, 'shifts', [0 0])
            %   files   .sbx path or cell of paths (concatenated, 'first' frames dropped each)
            %   shifts  [shiftOdd shiftEven] px applied while reading (to measure a residual)
            % B: lag, lagMedian, lagMAD, lagBands, bandCols, lags, cc, peakCorr,
            % atSearchLimit, reliable, channel, nFrames, rows (+ mode, applied for the caller)
            o = struct('channel', 1, 'crop', [1 80 Inf Inf], 'first', 1, 'nFrames', 200, ...
                'maxLag', 8, 'bands', 4, 'shifts', [0 0]);
            for k = 1:2:numel(varargin)
                o.(varargin{k}) = varargin{k+1};
            end
            files = cellstr(files);
            info = cellfun(@sbx_method.info, files, 'UniformOutput', false);
            info = [info{:}];
            crop = double(o.crop(:)');
            rc = [max(crop(1:2), 1), min(crop(3:4), [info(1).rows info(1).cols])];
            cum = [0 cumsum(max([info.nFramesDisk] - o.first, 0))];
            N = cum(end);
            idx = unique(round(linspace(1, N, min(o.nFrames, N))));
            Y = zeros(rc(3)-rc(1)+1, rc(4)-rc(2)+1, numel(idx), 'single');
            for f = 1:numel(info)
                sel = find(idx > cum(f) & idx <= cum(f+1));
                if ~isempty(sel)
                    Y(:, :, sel) = single(sbx_method.read(info(f).path, o.first + idx(sel) - cum(f) - 1, ...
                        'channel', o.channel, 'crop', rc, 'bidi', o.shifts, 'info', info(f)));
                end
            end
            [H, W, n] = size(Y);
            absRow = rc(1) - 1 + (1:H);                 % raw-frame row number of each row
            iA = find(mod(absRow, 2) == 1);             % odd raw rows ...
            iA = iA(iA < H);
            iB = iA + 1;                                % ... and the even row below each
            lags = -o.maxLag:o.maxLag;
            edges = round(linspace(0, W, o.bands + 1));
            if min(diff(edges)) <= 2 * o.maxLag
                error('bands too large: column bands narrower than 2*maxLag px.');
            end
            ccF = zeros(numel(lags), o.bands + 1, n);   % column 1 = all columns, then bands
            for f = 1:n
                A = Y(iA, :, f);
                Bm = Y(iB, :, f);
                for b = 0:o.bands
                    if b == 0, cols = 1:W; else, cols = edges(b)+1:edges(b+1); end
                    a = A(:, cols);
                    bb = Bm(:, cols);
                    ccF(:, b+1, f) = lagCorr(a - mean(a, 2), bb - mean(bb, 2), lags);
                end
            end
            cc = mean(ccF, 3);
            [lag, pk, atEdge] = peakLag(cc(:, 1), lags);
            lagBands = zeros(1, o.bands);
            for b = 1:o.bands
                lagBands(b) = peakLag(cc(:, b+1), lags);
            end
            lagF = zeros(1, n);
            for f = 1:n
                lagF(f) = peakLag(ccF(:, 1, f), lags);
            end
            B = struct('mode', '', 'lag', lag, 'applied', [0 0], ...
                'lagMedian', median(lagF), 'lagMAD', median(abs(lagF - median(lagF))), ...
                'lagBands', lagBands, 'bandCols', [edges(1:end-1)' + 1, edges(2:end)'] + rc(2) - 1, ...
                'lags', lags, 'cc', cc(:, 1), 'peakCorr', pk, 'atSearchLimit', atEdge, ...
                'reliable', ~atEdge && pk > 0.1, 'channel', o.channel, 'nFrames', n, 'rows', rc([1 3]));
        end

        function recs = findRecordings(dateDir, cfg)
            % Group the .sbx files by rid and pair each rid with its behavior file.
            % recs.ref / recs.refTif: reference .sbx paths (no extension) and output names.
            % recs.sessions: rid, name, sbx (paths, by segment), behavFile, refTif, refSbx.
            ridPos = cfg.RidPosition;
            segPos = 3 - ridPos;
            refRegex = ['^' strrep(cfg.refid, 'd', '\d') '$'];

            [allSbx, tok, other] = listMatching(dateDir, '*.sbx', '(\d{3})_(\d{3})\.sbx$', 2);
            if ~isempty(other)
                fprintf('  ! ignoring .sbx files not ending in ddd_ddd: %s\n', strjoin(other, ', '));
            end
            keep = hasKey(allSbx, cfg.recordingTag);
            if any(~keep)
                fprintf('  ! ignoring .sbx files without keyword %s: %s\n', ...
                    strjoin(cellstr(cfg.recordingTag), '/'), strjoin(allSbx(~keep), ', '));
            end
            names = allSbx(keep);
            tok = tok(keep, :);
            hasMat = isfile(fullfile(dateDir, regexprep(names, '\.sbx$', '.mat')));
            if any(~hasMat)
                warning('batch_2p_Calcium:missingSbxMat', 'Ignoring .sbx without accompanying .mat in %s: %s', ...
                    dateDir, strjoin(names(~hasMat), ', '));
                names = names(hasMat);
                tok = tok(hasMat, :);
            end
            if isempty(names)
                error('No .sbx files ending in ddd_ddd (with keyword %s) in %s', ...
                    strjoin(cellstr(cfg.recordingTag), '/'), dateDir);
            end
            [~, order] = sortrows(str2double(tok(:, [ridPos segPos])));   % by rid, then segment
            tok = tok(order, :);
            rids = tok(:, ridPos)';
            bases = reshape(fullfile(dateDir, regexprep(names(order), '\.sbx$', '')), 1, []);
            isRef = ~cellfun(@isempty, regexp(strcat(tok(:, 1), '_', tok(:, 2))', refRegex, 'once'));

            % behavior files: <anything>_<rid>_<yyyy.mm.dd>_<HH.MM.SS>.mat (not an .sbx info .mat)
            [mnames, btok] = listMatching(dateDir, '*.mat', ...
                '_(\d{3})_(\d{4}\.\d{2}\.\d{2})_\d{2}\.\d{2}\.\d{2}\.mat$', 2);
            isBeh = hasKey(mnames, cfg.behavTag) & ...
                ~ismember(regexprep(mnames, '\.mat$', ''), regexprep(allSbx, '\.sbx$', ''));
            mnames = mnames(isBeh);
            btok = btok(isBeh, :);

            % references, one per rid; output ref_field_MC.tif, or _rec<rid> if several
            [refRids, iu] = unique(rids(isRef), 'stable');
            recs.ref = bases(isRef);
            if numel(iu) < numel(recs.ref)
                fprintf('  ! several reference files share a rid; using the first of each\n');
            end
            recs.ref = recs.ref(iu);
            recs.refTif = repmat({'ref_field_MC.tif'}, size(recs.ref));
            if numel(recs.ref) > 1
                recs.refTif = strcat('ref_field_MC_rec', refRids, '.tif');
            end

            recs.sessions = struct('rid', {}, 'name', {}, 'sbx', {}, 'behavFile', {}, 'refTif', {}, 'refSbx', {});
            sessRids = unique(rids(~isRef));
            if isempty(sessRids)
                fprintf('  ! only reference recording(s) in this folder, no session to process\n');
            end
            for ri = 1:numel(sessRids)
                rid = sessRids{ri};
                cand = find(strcmp(btok(:, 1), rid));
                if numel(cand) > 1
                    [~, i] = min(cellfun(@numel, mnames(cand)));     % shortest name
                    fprintf('  ! rid %s: behavior files %s; using %s\n', rid, strjoin(mnames(cand), ', '), mnames{cand(i)});
                    cand = cand(i);
                end
                if isempty(cand)
                    fprintf('  ! rid %s: no behavior .mat with this rid (keyword %s), skipping\n', ...
                        rid, strjoin(cellstr(cfg.behavTag), '/'));
                    continue
                end
                segs = strsplit(mnames{cand}, '_');
                stem = strjoin(segs(1:find(strcmp(segs, rid), 1) - 1), '_');   % session folder: <stem>_rec<rid>
                % reference: same rid; else the only one; else the nearest rid
                refTif = '';
                refSbx = '';
                k = find(strcmp(refRids, rid));
                if isempty(k) && numel(refRids) == 1
                    k = 1;
                    fprintf('  o rid %s: no own reference; using the only one (rid %s)\n', rid, refRids{1});
                elseif isempty(k) && ~isempty(refRids)
                    [~, k] = min(abs(str2double(refRids) - str2double(rid)));
                    fprintf('  ! rid %s: no own reference; using the nearest (rid %s)\n', rid, refRids{k});
                end
                if ~isempty(k)
                    refTif = recs.refTif{k};
                    refSbx = recs.ref{k};
                end
                recs.sessions(end+1) = struct('rid', rid, 'name', [stem '_rec' rid], ...
                    'sbx', {bases(strcmp(rids, rid) & ~isRef)}, 'behavFile', fullfile(dateDir, mnames{cand}), ...
                    'refTif', refTif, 'refSbx', refSbx);
            end
            fprintf('  found %d session(s), %d reference file(s)\n', numel(recs.sessions), numel(recs.ref));
        end

        function movies = splitMovies(behavFile, ttlField)
            % Split behavior files <behav>_<ttlField>_movie<#>.mat, sorted by number.
            [fd, fb] = fileparts(behavFile);
            stem = [fb '_' ttlField '_movie'];
            [names, tok] = listMatching(fd, [stem '*.mat'], ['^' regexptranslate('escape', stem) '(\d+)\.mat$'], 1);
            movies = {};
            if isempty(names), return, end
            [~, order] = sort(str2double(tok));
            movies = reshape(fullfile(fd, names(order)), 1, []);
        end

        function [iM, iS] = matchMovies(nBeh, nSbx)
        % MATCHMOVIES  Pair behavior movies with .sbx segments when their
        % counts differ. Order is kept (no crossing), all of the shorter list is
        % paired, and the total |frame difference| is minimal (dynamic programming).
        %
        %   [iM, iS] = sbx_method.matchMovies(nBeh, nSbx)   % frame counts of each
        %   movie iM(k) is paired with sbx segment iS(k).
        swap = numel(nBeh) > numel(nSbx);        % true: the sbx list is the shorter one
        if swap
            a = nSbx; b = nBeh;
        else
            a = nBeh; b = nSbx;
        end
        K = numel(a);
        L = numel(b);
        D = inf(K + 1, L + 1);                   % D(i+1,j+1): best cost of a(1:i) within b(1:j)
        D(1, :) = 0;
        for i = 1:K
            for j = i:L
                D(i+1, j+1) = min(D(i+1, j), D(i, j) + abs(a(i) - b(j)));
            end
        end
        ib = zeros(1, K);                        % backtrack
        i = K;
        j = L;
        while i > 0
            if j > i && D(i+1, j+1) == D(i+1, j)
                j = j - 1;                       % b(j) left unpaired
            else
                ib(i) = j;
                i = i - 1;
                j = j - 1;
            end
        end
        if swap
            iM = ib;
            iS = 1:K;
        else
            iM = 1:K;
            iS = ib;
        end
        end

        function cb = concatBehavior(movies, nUsed, match)
        % CONCATBEHAVIOR  Stack the per-frame behavior of the split movies of one
        % session, aligned to the frames kept by sbx_motion_correct.
        %
        %   cb = sbx_method.concatBehavior(movies, nUsed, match)
        %   movies  cell of split behavior .mat files, one per sbx segment
        %   nUsed   imaging frames kept per segment (sbx frames - 1)
        %   match   struct array, one element per segment (copied to cb.segmentInfo)
        %
        % Per segment the first frame is dropped and the values are trimmed or padded
        % (with the last value) to nUsed. Added fields: segment, frameInSbx
        % (0-based frame in the .sbx), segmentInfo, and ignoreTrialActual (first and
        % last virmen trial of each segment, which are incomplete).
        nSeg = numel(movies);
        segs = cellfun(@load, movies, 'UniformOutput', false);
        isPerFrame = @(B, f) (isnumeric(B.(f)) || islogical(B.(f))) && isvector(B.(f)) ...
            && numel(B.(f)) == numel(B.timestamp);
        pf = cellfun(@(B) fieldnames(B)', segs, 'UniformOutput', false);
        pf = cellfun(@(B, fn) fn(cellfun(@(f) isPerFrame(B, f), fn)), segs, pf, 'UniformOutput', false);
        allPF = unique([pf{:}], 'stable');

        cb = struct();                           % not per-frame: first segment's copy
        for f = setdiff(fieldnames(segs{1})', [allPF, {'check_ttl_spacing'}], 'stable')
            cb.(f{1}) = segs{1}.(f{1});
        end
        for f = allPF                            % per-frame: drop first, trim / pad, stack
            parts = cell(nSeg, 1);
            for s = 1:nSeg
                v = [];
                if ismember(f{1}, pf{s})
                    v = double(segs{s}.(f{1})(2:end));
                end
                if isempty(v)
                    parts{s} = nan(nUsed(s), 1);
                else
                    parts{s} = reshape(v(min(1:nUsed(s), numel(v))), [], 1);
                end
            end
            cb.(f{1}) = vertcat(parts{:});
        end

        cb.segment = reshape(repelem((1:nSeg)', nUsed(:)), [], 1);
        cb.frameInSbx = cell2mat(arrayfun(@(n) (1:n)', nUsed(:), 'UniformOutput', false));
        ttl = cell(1, nSeg);
        for s = 1:nSeg
            ttl{s} = [];
            if isfield(segs{s}, 'check_ttl_spacing'), ttl{s} = segs{s}.check_ttl_spacing; end
        end
        [match.check_ttl_spacing] = ttl{:};
        cb.segmentInfo = match;
        if isfield(cb, 'virmen_trial')
            ign = [];
            for s = 1:nSeg
                v = cb.virmen_trial(cb.segment == s & ~isnan(cb.virmen_trial));
                if ~isempty(v)
                    ign = [ign, v(1), v(end)]; %#ok<AGROW>
                end
            end
            cb.ignoreTrialActual = unique(ign);
        end
        end

        function out = cropOut(tifPath)
        % CROPOUT  Output rectangle [row1 col1 row2 col2] (raw-frame pixels) of
        % a motion-corrected tif, from its _shifts.mat. [] if the file or the field is
        % missing (outputs made before it was saved).
        [d, n] = fileparts(tifPath);
        shiftsPath = fullfile(d, [n '_shifts.mat']);
        out = [];
        if ~isfile(shiftsPath) || ~ismember('crop', who('-file', shiftsPath))
            return
        end
        S = load(shiftsPath, 'crop');
        if isstruct(S.crop) && isfield(S.crop, 'out')
            out = double(S.crop.out(:)');
        end
        end

        function registerReference(refTif, mcPath, regrefPath)
        % REGISTERREFERENCE  Rigidly register the reference (structural channel) mean
        % image to the session mean image; the channels differ, so a multimodal metric
        % is used.
        %
        %   sbx_method.registerReference(refTif, mcPath, regrefPath)
        %   refTif      motion-corrected reference tif ('' = no reference: skipped)
        %   mcPath      ConcatenatedMovie_MC.tif of the session
        %   regrefPath  output: registered reference tif; the session mean / max
        %               images, all_registeration_imgs.mat and merged_channel.tif
        %               (green = session, red = reference) go in the session folder
        if isempty(refTif)
            fprintf('  ! no reference for this session; skipping registration\n');
            return
        end
        shiftsOf = @(t) regexprep(t, '\.tif$', '_shifts.mat');
        if ~isfile(shiftsOf(refTif)) || ~isfile(shiftsOf(mcPath))
            fprintf('  ! reference or session motion correction missing; skipping registration\n');
            return
        end
        sessDir = fileparts(mcPath);
        R = load(shiftsOf(refTif), 'meanImg');
        S = load(shiftsOf(mcPath), 'meanImg', 'maxImg');
        avgRef = double(R.meanImg);
        avgMc = double(S.meanImg);
        maxMc = double(S.maxImg);
        [optimizer, metric] = imregconfig('multimodal');

        % start from the translation implied by the two output rectangles (raw px):
        % reference pixel (r, c) is session pixel (r + dy, c + dx); zero if equal
        refOut = sbx_method.cropOut(refTif);
        sessOut = sbx_method.cropOut(mcPath);
        init = [];
        if numel(refOut) == 4 && numel(sessOut) == 4
            d = refOut(1:2) - sessOut(1:2);
            if isequal(refOut, sessOut)
                fprintf('  reference and session cover the same pixels (%d x %d)\n', size(avgMc));
            else
                fprintf('  reference is offset by [dy dx] = [%d %d] px from the session\n', d(1), d(2));
            end
            try
                init = rigid2d(0, [d(2) d(1)]);              % R2022b+
            catch
                init = affine2d([1 0 0; 0 1 0; d(2) d(1) 1]);
            end
        else
            fprintf('  ! no crop info in the shifts files: registering from image content only\n');
        end
        registered = [];
        if ~isempty(init)
            try
                registered = imregister(avgRef, avgMc, 'rigid', optimizer, metric, 'InitialTransformation', init);
            catch ME
                fprintf('  ! registration with the crop-based start failed (%s); retrying without\n', ME.message);
            end
        end
        if isempty(registered)
            registered = imregister(avgRef, avgMc, 'rigid', optimizer, metric);
        end

        writeTif16(regrefPath, registered);
        writeTif16(fullfile(sessDir, 'ConcatenatedMovie_MC_avg_registered.tif'), avgMc);
        writeTif16(fullfile(sessDir, 'ConcatenatedMovie_MC_max_registered.tif'), maxMc);
        avg_mcvid = avgMc; avg_refvid = avgRef; max_mcvid = maxMc;    % names kept for later steps
        save(fullfile(sessDir, 'all_registeration_imgs.mat'), 'registered', 'avg_mcvid', 'avg_refvid', 'max_mcvid');
        merged = imfuse(avgMc, registered, 'falsecolor', 'Scaling', 'joint', 'ColorChannels', [2 1 0]);
        imwrite(merged, fullfile(sessDir, 'merged_channel.tif'));
        end

    end
end


%% Local functions (used by the methods above)

function Y = shiftCols(Y, s)
% Move the content of every row by s columns (+ = right): out(:,c) = in(:,c-s);
% Lanczos-3 interpolation, edge pixels replicated (same as sbx_motion_correct).
if s == 0
    return
end
nC = size(Y, 2);
k = floor(s);
f = s - k;
j = -2:3;
w = lanczos3(j - f);
w = w / sum(w);
cols = 1:nC;
Z = zeros(size(Y), 'like', Y);
for t = 1:numel(j)
    if w(t) == 0
        continue
    end
    Z = Z + w(t) * Y(:, min(max(cols - k - j(t), 1), nC), :);
end
Y = Z;
end

function w = lanczos3(d)
w = zeros(size(d));
m = abs(d) < 3 & d ~= 0;
dm = d(m);
w(m) = 3 * sin(pi*dm) .* sin(pi*dm/3) ./ (pi^2 * dm.^2);
w(d == 0) = 1;
end

function c = lagCorr(a, b, lags)
% normalised correlation of a(:, c + l) with b(:, c) for every lag l
c = zeros(numel(lags), 1);
for i = 1:numel(lags)
    l = lags(i);
    if l >= 0
        x = a(:, 1+l:end);
        y = b(:, 1:end-l);
    else
        x = a(:, 1:end+l);
        y = b(:, 1-l:end);
    end
    d = sqrt(sum(x(:).^2) * sum(y(:).^2));
    if d > 0
        c(i) = sum(x(:) .* y(:)) / d;
    end
end
end

function [lag, pk, atEdge] = peakLag(cc, lags)
% sub-pixel (parabolic) position of the correlation maximum
[pk, im] = max(cc);
lag = lags(im);
atEdge = im == 1 || im == numel(lags);
if ~atEdge
    den = cc(im-1) - 2*cc(im) + cc(im+1);
    if den < 0
        lag = lag + 0.5 * (cc(im-1) - cc(im+1)) / den;
    end
end
end

function [names, tok, other] = listMatching(folder, pattern, regex, nTok)
    % Files matching the dir pattern whose names match regex; tok = regex tokens.
    dd = dir(fullfile(folder, pattern));
    allNames = {dd.name};
    t = regexp(allNames, regex, 'tokens', 'once');
    ok = ~cellfun(@isempty, t);
    names = allNames(ok);
    other = allNames(~ok);
    tok = cell(0, nTok);
    if any(ok)
        tok = vertcat(t{ok});
    end
end

function tf = hasKey(names, keys)
    % True for names containing any keyword (case-insensitive); all if keys is empty.
    tf = true(size(names));
    if ~isempty(keys) && ~isempty(names)
        tf = contains(names, cellstr(keys), 'IgnoreCase', true);
    end
end

function writeTif16(path, img)
t = Tiff(path, 'w');
c = onCleanup(@() t.close());
t.setTag(struct('ImageLength', size(img, 1), 'ImageWidth', size(img, 2), ...
    'Photometric', Tiff.Photometric.MinIsBlack, 'BitsPerSample', 16, ...
    'SamplesPerPixel', 1, 'SampleFormat', Tiff.SampleFormat.UInt, ...
    'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky, ...
    'Compression', Tiff.Compression.None));
t.write(uint16(img));
end
