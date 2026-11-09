% batch_2p_Calcium.m
%
% Batch preprocessing of Scanbox 2p calcium imaging + behavior, per mouse and
% imaging date. Motion correction reads the .sbx directly (sbx_motion_correct).
% Save your own copy and edit USER INPUT. A step is skipped if its output
% exists (unless cfg.overwrite); a failing session is reported and the rest go on.
%
% DATA LAYOUT (one imaging day per folder)
%   rootDir\<mouse>\<yymmdd>\
%       <name>_000_000.sbx/.mat    rid 000 = structural (red) reference
%       <name>_001_000.sbx/.mat    rid 001, segment 0   (_001_001 = segment 1 ...)
%       <mid>_<exptype>_rec_001.mat  behavior file of rid 001: ttlIn2 holds one
%                                  TTL per imaging frame, one TTL bout per segment
%
% STEPS (per date)
%   sessions   split & bin behavior -> match movies to sbx segments ->
%              motion-correct all segments together (green) -> concatenate behavior
%   reference  motion-correct the red reference once, after the sessions, cropped to
%              the session's output rectangle (so no frame is corrected twice)
%   register   reference mean image -> session mean image
%   cfg.diagnostics (optional, slower): raw segment projections before the
%              correction; PC1, mean F, shifts, frame correlation, mean-image
%              and zoom comparisons, scan-phase offset, kymographs drawn inside
%              the correction (help mc_diag_method.session); reference registration figure
%
% OUTPUTS
%   <date>\ref_field_MC.tif (+ _shifts.mat)     motion-corrected reference
%   <date>\<behavior>_ttlIn2_movie#.mat         split behavior, one per segment
%   <date>\file_locations.mat, matchedBeh_Tif_info.mat
%   <date>\<session>_segment_projections.png    (diagnostics)
%   <date>\<exptype>_rec<rid>\
%       ConcatenatedMovie_MC.tif                uint16, all segments
%       ConcatenatedMovie_MC_shifts.mat         shifts, mean/max image, template,
%                                               crop, 'bidi' scan-phase record
%       output_behaviorConcat.mat               behavior aligned to the movie rows
%       ref_field_MC_registeration.tif, ConcatenatedMovie_MC_avg/max_registered.tif,
%       all_registeration_imgs.mat, merged_channel.tif
%       mc_diagnostics\                         figures (diagnostics)
%
% Code: sbx_method (everything done on the .sbx / its outputs), mc_diag_method
% (figures and checks), mc_function\sbx_motion_correct (this folder).
% Also needs split_and_bin_grid_behav_files, check_TTL_spacing (DataProcessing),
% vec.m (CVX / SeDuMi), Image Processing Toolbox.

%%%%%%%%%%%%%%%%%%
%%% USER INPUT %%%
%%%%%%%%%%%%%%%%%%
clear
cfg.rootDir = 'O:\2p_animals\';

% which folders: mouseList + dateList -> rootDir\<mouse>\<date>; only mouseList ->
% every folder below rootDir\<mouse> holding .sbx files; neither -> all of rootDir
cfg.mouseList   = {'mouse1'};
cfg.dateList    = {};
cfg.excludeList = {};   % folders to skip, relative to rootDir ('mouse3' or 'mouse1\260930')

cfg.codePaths = {
    'C:\Users\HoweLab\Documents\GitHub\Calcium_2p_preprocessing'
    'C:\Users\HoweLab\Documents\GitHub\DataProcessing'
    };

% steps
cfg.steps.exportTifs     = false;  % per-channel raw tifs of every .sbx (inspection only, slow)
cfg.steps.splitBehavior  = true;
cfg.steps.motionCorrect  = true;
cfg.steps.behaviorConcat = true;
cfg.steps.reference      = true;   % red reference, after the sessions
cfg.steps.registerRef    = true;
cfg.diagnostics = true;           % true: segment projections + figures inside the motion correction
cfg.diagOpts    = {};              % options of mc_diagnostics, e.g. {'nSample', 1000, 'nPC', 10}
cfg.overwrite   = true;           % true = redo steps whose outputs exist

% recordings: file names end in ddd_ddd; position RidPosition (1 or 2) is the recording id
cfg.RidPosition  = 1;
cfg.refid        = 'ddd_000';      % the reference .sbx ('d' = any digit)
cfg.refChannel   = 2;              % PMT channel of the reference (2 = red)
cfg.funcChannel  = 1;              % PMT channel of the functional movie (1 = green)
cfg.behavTag     = {};             % keyword(s) the behavior file name must contain
cfg.recordingTag = {};             % keyword(s) the .sbx file name must contain

% behavior split & bin
cfg.ttlField         = 'ttlIn2';
cfg.splitArgs        = {};         % extra inputs, e.g. {'virmenStartTTL', 3}
cfg.maxFrameMismatch = 5;          % max |behavior frames - sbx frames| per segment

% motion correction (help sbx_motion_correct). Do not set outfile, channel,
% dropFirstFrame, crop, outCrop here. bidiShift 'auto' corrects the even/odd
% scan-line offset of every .sbx; the estimate is saved as 'bidi'.
cfg.mcOpts = {'maxShift', 0.1, 'nonrigid', true, 'bidiShift', 'auto'};

% raw-frame crop [row1 col1 row2 col2]; columns 1-79 are the scan turnaround
cfg.crop = [1 80 Inf Inf];

% Output rectangle: each run keeps the region valid for its own shifts. The
% reference is cropped to the session's rectangle (matchRefCrop), or give one
% fixed rectangle [row1 col1 row2 col2] (raw-frame px, inside cfg.crop) for all runs.
cfg.matchRefCrop = true;
cfg.outCrop      = [];
%%%%%%%%%%%%%%%%%%%%%%%
%%% END: USER INPUT %%%
%%%%%%%%%%%%%%%%%%%%%%%

%% SETUP
close all
found = isfolder(cfg.codePaths);
cellfun(@(x) addpath(genpath(x)), cfg.codePaths(found));
if ~all(found)
    warning('Code path(s) not found: %s', strjoin(cfg.codePaths(~found), ', '));
end
required = {'sbx_motion_correct', 'sbx_method', 'mc_diag_method', 'split_and_bin_grid_behav_files', ...
    'check_TTL_spacing', 'vec'};
missing = required(cellfun(@(f) exist(f, 'file') == 0, required));
if ~isempty(missing)
    error('Functions not on the path: %s (vec.m comes with CVX / SeDuMi).', strjoin(missing, ', '));
end
reserved = {'outfile', 'channel', 'dropFirstFrame', 'crop', 'outCrop', 'diagnostics', 'diagDir', 'diagOpts'};
if any(ismember(lower(cfg.mcOpts(1:2:end)), lower(reserved)))
    error('mcOpts must not set %s; the batch script sets them.', strjoin(reserved, ', '));
end
warning('off', 'MATLAB:polyshape:repairedBySimplify')
fprintf('batch_2p_Calcium started %s\nrootDir: %s\n', datestr(now), cfg.rootDir);
if isempty(gcp('nocreate')), parpool('Threads'); end

% folders to process
dateDirs = {};
if ~isempty(cfg.mouseList) && ~isempty(cfg.dateList)
    [mm, dd] = ndgrid(1:numel(cfg.mouseList), 1:numel(cfg.dateList));
    dateDirs = arrayfun(@(i, j) fullfile(cfg.rootDir, cfg.mouseList{i}, cfg.dateList{j}), ...
        mm(:), dd(:), 'UniformOutput', false)';
    miss = ~isfolder(dateDirs);
    if any(miss)
        fprintf('! date folder(s) not found, skipping: %s\n', strjoin(dateDirs(miss), ', '));
        dateDirs = dateDirs(~miss);
    end
else
    searchRoots = {cfg.rootDir};
    if ~isempty(cfg.mouseList)
        searchRoots = fullfile(cfg.rootDir, cfg.mouseList);
    end
    for r = 1:numel(searchRoots)
        dd = dir(fullfile(searchRoots{r}, '**', '*.sbx'));
        hit = sort(unique({dd(~[dd.isdir]).folder}));
        fprintf('%s: %d folder(s) with .sbx files\n', searchRoots{r}, numel(hit));
        dateDirs = [dateDirs, hit]; %#ok<AGROW>
    end
    dateDirs = unique(dateDirs, 'stable');
    if ~isempty(cfg.dateList)
        [~, leaf] = cellfun(@fileparts, dateDirs, 'UniformOutput', false);
        dateDirs = dateDirs(ismember(leaf, cfg.dateList));
    end
end
if ~isempty(cfg.excludeList) && ~isempty(dateDirs)
    rootNorm = lower(strip(cfg.rootDir, 'right', filesep));
    rel = lower(strip(erase(lower(dateDirs), rootNorm), 'left', filesep));
    isExcl = false(size(dateDirs));
    for e = reshape(lower(strip(cfg.excludeList, 'both', filesep)), 1, [])
        isExcl = isExcl | strcmp(rel, e{1}) | startsWith(rel, [e{1} filesep]);
    end
    if any(isExcl)
        fprintf('Excluded %d folder(s): %s\n', nnz(isExcl), strjoin(dateDirs(isExcl), ', '));
    end
    dateDirs = dateDirs(~isExcl);
end

if isempty(dateDirs)
    warning('No folders with .sbx files found under %s.', cfg.rootDir);
end

%% MAIN LOOP
summary = cell(0, 4);                 % {mouse, date, session, status}
for d = 1:numel(dateDirs)
    [parent, dateStr] = fileparts(dateDirs{d});
    label = strip(erase(parent, strip(cfg.rootDir, 'right', filesep)), 'left', filesep);
    fprintf('\n====================%s: %s ====================\n', label, dateStr);
    rows = cell(0, 2);                % {session, status}

    [status, recs] = attempt(@() sbx_method.findRecordings(dateDirs{d}, cfg));
    if ~strcmp(status, 'ok')
        summary = [summary; {label, dateStr, '-', status}]; %#ok<AGROW>
        continue
    end
    if cfg.steps.exportTifs
        allSbx = [recs.ref, recs.sessions.sbx];
        for k = 1:numel(allSbx)
            status = attempt(@() sbx_method.toTif(allSbx{k}, 'crop', cfg.crop, ...
                'outDir', fullfile(dateDirs{d}, 'raw_tifs'), 'overwrite', cfg.overwrite));
            if ~strcmp(status, 'ok')
                rows(end+1, :) = {['tif ' nameOf(allSbx{k})], status}; %#ok<SAGROW>
            end
        end
    end

    % sessions first (functional channel): the reference is cropped to their output
    file_locs = struct([]);
    matchedBeh_Tif = cell(0, 4);
    sessOK = false(1, numel(recs.sessions));
    sessRow = zeros(1, numel(recs.sessions));
    for s = 1:numel(recs.sessions)
        sess = recs.sessions(s);
        fprintf('\n--- session %s (rid %s, %d segment(s)) ---\n', sess.name, sess.rid, numel(sess.sbx));
        [status, fl, matchRows] = attempt(@() processSession(dateDirs{d}, sess, cfg));
        rows(end+1, :) = {sess.name, status}; %#ok<SAGROW>
        sessRow(s) = size(rows, 1);
        if strcmp(status, 'ok')
            sessOK(s) = true;
            file_locs = [file_locs fl]; %#ok<AGROW>
            matchedBeh_Tif = [matchedBeh_Tif; matchRows]; %#ok<AGROW>
        end
    end

    if cfg.steps.reference
        fprintf('\n--- reference ---\n');
        status = attempt(@() processReference(dateDirs{d}, recs, cfg));
        if ~strcmp(status, 'ok')
            rows(end+1, :) = {'reference', status}; %#ok<SAGROW>
        end
    end

    for s = find(sessOK)
        status = attempt(@() registerSession(dateDirs{d}, recs.sessions(s), cfg));
        if ~strcmp(status, 'ok')
            rows{sessRow(s), 2} = ['registration ' status];
        end
    end
    for i = 1:numel(file_locs)        % absolute path of the registered reference, '' if none
        p = fullfile(dateDirs{d}, file_locs(i).rec_dname, 'ref_field_MC_registeration.tif');
        if isfile(p), file_locs(i).regref_path = p; end
    end
    if ~isempty(file_locs)
        save(fullfile(dateDirs{d}, 'file_locations.mat'), 'file_locs');
        save(fullfile(dateDirs{d}, 'matchedBeh_Tif_info.mat'), 'matchedBeh_Tif');
    end
    summary = [summary; repmat({label, dateStr}, size(rows, 1), 1), rows]; %#ok<AGROW>
end

fprintf('\n==================== SUMMARY ====================\n');
for i = 1:size(summary, 1)
    fprintf('%-10s %-8s %-22s %s\n', summary{i, :});
end
fprintf('batch_2p_Calcium finished %s\n', datestr(now));


%% LOCAL FUNCTIONS

function [fl, matchRows] = processSession(dateDir, sess, cfg)
    % Split behavior, match it to the segments, motion-correct, concatenate
    % behavior. Returns the file_locs entry and the matchedBeh_Tif rows.
    sessDir = fullfile(dateDir, sess.name);
    if ~isfolder(sessDir), mkdir(sessDir); end
    mcPath = fullfile(sessDir, 'ConcatenatedMovie_MC.tif');
    catPath = fullfile(sessDir, 'output_behaviorConcat.mat');
    nSeg = numel(sess.sbx);

    % split & bin behavior
    movies = sbx_method.splitMovies(sess.behavFile, cfg.ttlField);
    if shouldRun(cfg.steps.splitBehavior, numel(movies) == nSeg, cfg, 'split & bin behavior')
        [~, retry] = split_and_bin_grid_behav_files(sess.behavFile, cfg.splitArgs{:});
        if any(retry)
            fprintf('  ! split_and_bin flagged a likely virmen misalignment; check ''virmenStartTTL'' in splitArgs\n');
        end
        movies = sbx_method.splitMovies(sess.behavFile, cfg.ttlField);
    end

    % match behavior movies to sbx segments
    nSbx = cellfun(@(f) getfield(sbx_method.info(f), 'nFramesDisk'), sess.sbx);
    nBeh = cellfun(@(f) numel(getfield(load(f, 'timestamp'), 'timestamp')), movies);
    fprintf('  segment  sbx frames  behavior frames\n');
    for k = 1:max(nSeg, numel(movies))
        fprintf('  %7d  %10s  %15s\n', k, numOrDash(nSbx, k), numOrDash(nBeh, k));
    end
    if numel(movies) ~= nSeg
        warning('%d behavior movie(s) (TTL bouts) but %d sbx segment(s); matching existing movies...', numel(movies), nSeg);
        [iM, iS] = sbx_method.matchMovies(nBeh, nSbx);
        for k = 1:numel(iM)
            fprintf('    movie%d (%d fr) -> %s (%d fr)\n', iM(k), nBeh(iM(k)), nameOf(sess.sbx{iS(k)}), nSbx(iS(k)));
        end
        if nSeg > numel(iS)
            fprintf('  ! sbx without behavior, left out: %s\n', strjoin(nameOf(sess.sbx(setdiff(1:nSeg, iS))), ', '));
        end
        if numel(movies) > numel(iM)
            fprintf('  ! behavior movies without sbx, ignored: %s\n', ...
                strjoin(nameOf(movies(setdiff(1:numel(movies), iM))), ', '));
        end
        movies = movies(iM);
        nBeh = nBeh(iM);
        sess.sbx = sess.sbx(iS);          % motion correction uses only matched segments
        nSbx = nSbx(iS);
    end
    frameDiff = nBeh - nSbx;
    if any(abs(frameDiff) > cfg.maxFrameMismatch)
        error('Behavior and sbx frame counts differ by more than %d in segment(s) %s.', ...
            cfg.maxFrameMismatch, mat2str(find(abs(frameDiff) > cfg.maxFrameMismatch)));
    end
    sbxNames = strcat(nameOf(sess.sbx), '.sbx');
    match = struct('sbx', sbxNames, 'behavMovie', nameOf(movies), 'nFramesSbx', num2cell(nSbx), ...
        'nFramesBehav', num2cell(nBeh), 'frameDiff', num2cell(frameDiff));
    matchRows = [strcat(sess.sbx(:), '.sbx'), movies(:), num2cell(nSbx(:)), num2cell(nBeh(:))];

    % diagnostics before the correction: raw projections of the segments to be concatenated
    segPng = fullfile(dateDir, [sess.name '_segment_projections.png']);
    if shouldRun(cfg.diagnostics, {segPng}, cfg, 'segment projections')
        try
            mc_diag_method.segments(sess.sbx, segPng, 'channel', cfg.funcChannel, 'crop', cfg.crop, 'title', sess.name);
        catch ME
            warning('batch_2p_Calcium:diag', 'Segment projections failed: %s', ME.message);
        end
    end

    % motion correction (diagnostic figures are drawn inside it)
    if shouldRun(cfg.steps.motionCorrect, {mcPath, shiftsFile(mcPath)}, cfg, 'motion correction')
        opts = [cfg.mcOpts, forcedCrop(cfg)];
        if cfg.diagnostics
            opts = [opts, {'diagnostics', true, 'diagDir', fullfile(sessDir, 'mc_diagnostics'), ...
                'diagOpts', [{'name', sess.name}, cfg.diagOpts]}];
        end
        sbx_motion_correct(sess.sbx, 'outfile', mcPath, 'channel', cfg.funcChannel, ...
            'dropFirstFrame', true, 'crop', cfg.crop, opts{:});
    end
    if cfg.diagnostics && isfile(mcPath) && ~isfolder(fullfile(sessDir, 'mc_diagnostics'))
        fprintf(['  ! no diagnostic figures: the motion correction already existed. Run ' ...
            'mc_diag_method.session(mcPath, sbxFiles, outDir) separately, or set cfg.overwrite.\n']);
    end
    bidiInfo = readBidi(shiftsFile(mcPath));
    if ~isempty(bidiInfo)
        fprintf('  o scan-phase offset %+.2f px, applied %+.2f px, residual %+.2f px\n', ...
            bidiInfo.lag, bidiInfo.appliedLag, bidiInfo.lagAfter);
    elseif isfile(shiftsFile(mcPath)) && any(strcmpi(cfg.mcOpts(1:2:end), 'bidiShift'))
        fprintf('  ! existing motion correction has no scan-phase record (made without bidiShift)\n');
    end

    % behavior aligned to the motion-corrected movie
    nUsed = nSbx - 1;                     % the first frame of every segment is dropped
    if shouldRun(cfg.steps.behaviorConcat, {catPath}, cfg, 'concatenate behavior')
        if isfile(shiftsFile(mcPath))
            nMovie = size(getfield(load(shiftsFile(mcPath), 'frameIndex'), 'frameIndex'), 1);
            if nMovie ~= sum(nUsed)
                error('Motion-corrected movie has %d frames but the sbx segments give %d.', nMovie, sum(nUsed));
            end
        end
        concatenatedBehavior = sbx_method.concatBehavior(movies, nUsed, match);
        save(catPath, 'concatenatedBehavior');
    end

    fl = struct('rec_dname', sess.name, ...
        'ref_fname', sess.refTif, ...                 % '' if none
        'behav_fname', [nameOf(sess.behavFile) '.mat'], ...
        'rec_fnames', {sbxNames}, ...
        'snb_behavs', {strcat(nameOf(movies), '.mat')}, ...
        'mc_path', fullfile(sess.name, 'ConcatenatedMovie_MC.tif'), ...
        'catbehav_path', fullfile(sess.name, 'output_behaviorConcat.mat'), ...
        'regref_path', '', ...                        % absolute, set once registered
        'mc_shifts_path', fullfile(sess.name, 'ConcatenatedMovie_MC_shifts.mat'), ...
        'rid', sess.rid, ...
        'bidi', bidiInfo, ...                         % [] for older motion correction
        'match', match);
end

function processReference(dateDir, recs, cfg)
    % Motion-correct each reference (red) after the sessions, with the output
    % rectangle of the session(s) that use it. A reference that exists with
    % another rectangle is redone (it is short; sessions are never redone).
    if isempty(recs.ref)
        fprintf('  ! no reference recording (refid %s)\n', cfg.refid);
        return
    end
    for k = 1:numel(recs.ref)
        refOut = fullfile(dateDir, recs.refTif{k});
        target = cfg.outCrop(:)';                     % a fixed rectangle wins
        fromSession = false;
        if isempty(target) && cfg.matchRefCrop
            target = sessionCrop(dateDir, recs, recs.refTif{k});
            fromSession = ~isempty(target);
            if isempty(target)
                fprintf('  ! no session output rectangle for %s; reference keeps its own crop\n', recs.refTif{k});
            end
        end
        done = isfile(refOut) && isfile(shiftsFile(refOut));
        if done && ~isempty(target)
            done = isequal(sbx_method.cropOut(refOut), target);
        end
        if ~shouldRun(true, done, cfg, ['reference motion correction ' nameOf(recs.ref{k})])
            continue
        end
        forced = {};
        if ~isempty(target)
            forced = {'outCrop', target};
            fprintf('  o output rectangle [%s]\n', num2str(target));
        end
        args = {'outfile', refOut, 'channel', cfg.refChannel, 'dropFirstFrame', true, 'crop', cfg.crop};
        try
            sbx_motion_correct(recs.ref{k}, args{:}, forced{:}, cfg.mcOpts{:});
        catch ME
            if ~fromSession, rethrow(ME); end
            warning('batch_2p_Calcium:refCrop', ...
                'Reference with the session crop failed (%s); using its own crop.', ME.message);
            sbx_motion_correct(recs.ref{k}, args{:}, cfg.mcOpts{:});
        end
    end
end

function c = sessionCrop(dateDir, recs, refTif)
    % Output rectangle of the session(s) using this reference (the
    % intersection if several); [] if none has crop info.
    c = [];
    for s = 1:numel(recs.sessions)
        if ~strcmp(recs.sessions(s).refTif, refTif), continue, end
        r = sbx_method.cropOut(fullfile(dateDir, recs.sessions(s).name, 'ConcatenatedMovie_MC.tif'));
        if isempty(r), continue, end
        if isempty(c), c = r; else, c = [max(c(1:2), r(1:2)), min(c(3:4), r(3:4))]; end
    end
    if ~isempty(c) && (c(3) < c(1) || c(4) < c(2))
        warning('batch_2p_Calcium:refCrop', 'Sessions sharing %s have no common output rectangle.', refTif);
        c = [];
    end
end

function registerSession(dateDir, sess, cfg)
    % Register the reference to the session (needs the corrected reference).
    sessDir = fullfile(dateDir, sess.name);
    mcPath = fullfile(sessDir, 'ConcatenatedMovie_MC.tif');
    regrefPath = fullfile(sessDir, 'ref_field_MC_registeration.tif');
    refTif = '';
    if ~isempty(sess.refTif), refTif = fullfile(dateDir, sess.refTif); end
    fprintf('  session %s:\n', sess.name);
    if shouldRun(cfg.steps.registerRef, {regrefPath}, cfg, 'register reference')
        sbx_method.registerReference(refTif, mcPath, regrefPath);
    end
    regPng = fullfile(sessDir, 'mc_diagnostics', '09_reference_registration.png');
    if cfg.diagnostics && ~isempty(refTif) && isfile(fullfile(sessDir, 'all_registeration_imgs.mat')) ...
            && (cfg.overwrite || ~isfile(regPng))
        try
            mc_diag_method.referenceFigure(refTif, mcPath, regPng, sess.name);
        catch ME
            warning('batch_2p_Calcium:diag', 'Reference registration figure failed: %s', ME.message);
        end
    end
end

function b = readBidi(shiftsPath)
    % Summary of the 'bidi' record of a _shifts.mat; [] if there is none.
    b = [];
    if ~isfile(shiftsPath) || ~ismember('bidi', who('-file', shiftsPath))
        return
    end
    B = getfield(load(shiftsPath, 'bidi'), 'bidi');
    b = struct('mode', B.mode, 'lag', B.lag, 'appliedLag', B.applied(2) - B.applied(1), ...
        'lagAfter', B.lagAfter, 'lagBands', B.lagBands, 'reliable', B.reliable);
end

function c = forcedCrop(cfg)
    % {'outCrop', rect} if cfg.outCrop is set
    c = {};
    if ~isempty(cfg.outCrop)
        c = {'outCrop', cfg.outCrop};
    end
end

function varargout = attempt(fn)
    % Run fn; first output 'ok' or 'FAILED: <message>', then fn's outputs.
    varargout = cell(1, max(nargout, 1));
    try
        if nargout <= 1
            fn();
        else
            [varargout{2:nargout}] = fn();
        end
        varargout{1} = 'ok';
    catch ME
        fprintf(2, '  ! ERROR: %s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
        varargout{1} = ['FAILED: ' ME.message];
    end
end

function run = shouldRun(enabled, done, cfg, label)
    % True if the step is on and (overwrite or not done). done: logical or
    % a cell of output files that must all exist.
    if iscell(done)
        done = all(isfile(done));
    end
    run = enabled && (cfg.overwrite || ~done);
    if run
        fprintf('  o %s\n', label);
    elseif enabled
        fprintf('  o %s: already done, skipping\n', label);
    end
end

function p = shiftsFile(tifPath)
    [d, n] = fileparts(tifPath);
    p = fullfile(d, [n '_shifts.mat']);
end

function n = nameOf(paths)
    % File name without folder or extension (char or cell).
    [~, n] = cellfun(@fileparts, cellstr(paths), 'UniformOutput', false);
    if ischar(paths), n = n{1}; end
end

function s = numOrDash(v, k)
    s = '-';
    if k <= numel(v), s = num2str(v(k)); end
end
