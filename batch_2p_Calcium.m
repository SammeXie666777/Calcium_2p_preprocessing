% batch_2p_Calcium.m
%
% Batch preprocessing of Scanbox 2p calcium imaging + behavior, per
% mouse and recording date. Replaces batch_2p_processData_Lily.m:
% motion correction reads .sbx directly (sbx_motion_correct), so there is
% no sbx -> tif conversion step.
%
% Save your own copy of this script and edit the USER INPUT section.
%
% =========================================================================
% RAW DATA LAYOUT (everything for one imaging day flat in the date folder)
% =========================================================================
%   rootDir\<mouse>\<yymmdd>\
%       yymmdd_000_000.sbx + .mat      rid 000 = red structural reference
%       yymmdd_001_000.sbx + .mat      rid 001, segment 0
%       yymmdd_001_001.sbx + .mat      rid 001, segment 1
%       <mid>_<exptype>_rec_001.mat    behavior file for rid 001
%       ...
%   - Scanbox names are <Animal>_<Unit>_<Exp>: set Animal = date (6 digits),
%     Unit = recording id (rid, 3 digits), Exp = segment index (3 digits).
%   - The behavior file is matched to the imaging by rid ('rec_<rid>').
%     <mid> is 3 digits. It must contain ttlIn2 (or ttlIn_Scanbox), one TTL
%     per imaging frame; each bout of TTLs is one Scanbox segment.
%   - Date folders must be 6 digits and start with one of datePrefixes.
%
% =========================================================================
% STEPS (per date; each step is skipped if its output exists, unless
% overwrite = true; a failing session is reported and the rest continue)
% =========================================================================
%   0. Find recordings: group .sbx segments by rid, pair each rid with its
%      behavior file.
%   1. Reference: motion-correct the rid 000 red channel.
%   2. Per session (rid):
%      a. Split & bin behavior into one file per TTL bout
%         (split_and_bin_grid_behav_files).
%      b. Match: #behavior movies must equal #sbx segments; per segment,
%         |behavior frames - sbx frames| must be <= maxFrameMismatch.
%      c. Motion correction of all segments together (sbx_motion_correct).
%      d. Concatenate behavior aligned to the motion-corrected movie: per
%         segment, drop the first frame (the imaging's first frame is
%         dropped too), trim or pad (with the last value) to the number of
%         imaging frames, then stack segments.
%      e. Register the reference mean image to the session mean image.
%
% =========================================================================
% OUTPUTS (same names as batch_2p_processData_Lily, so later steps work)
% =========================================================================
%   <date>\
%     ref_field_MC.tif, ref_field_MC_shifts.mat     step 1
%     <behavior>_ttlIn2_movie#.mat, *_movieBoundaries.fig    step 2a
%     file_locations.mat       file_locs struct array, one per session
%     matchedBeh_Tif_info.mat  {sbx, behavior movie, #sbx frames, #behav frames}
%     <exptype>_rec<rid>\
%       ConcatenatedMovie_MC.tif          step 2c (uint16, all segments)
%       ConcatenatedMovie_MC_shifts.mat   shifts, mean/max image, template
%       output_behaviorConcat.mat         step 2d, 'concatenatedBehavior':
%           per-frame fields  one row per frame of ConcatenatedMovie_MC.tif
%           segment           which sbx segment each frame came from
%           frameInSbx        0-based frame index within that .sbx
%           segmentInfo       per segment: files, frame counts, mismatch,
%                             check_ttl_spacing
%           ignoreTrialActual first/last virmen trial of each segment
%       ref_field_MC_registeration.tif    step 2e
%       ConcatenatedMovie_MC_avg_registered.tif / _max_registered.tif
%       all_registeration_imgs.mat, merged_channel.tif
%
% Requires: sbx_motion_correct.m, sbx_file_info.m (this folder);
% split_and_bin_grid_behav_files.m, check_TTL_spacing.m (HoweLab
% DataProcessing); vec.m (CVX / SeDuMi); Image Processing Toolbox for step 2e.

% Folders to process ("date" folders = where segments are concatenated):
%   mouseList + dateList given -> rootDir\<mouse>\<date> for every pair
%   only mouseList given       -> every folder under rootDir\<mouse> that
%                                 directly holds .sbx files (any depth)
%   neither given              -> every folder under rootDir that directly
%                                 holds .sbx files (any depth)
% Each such folder is treated as one date; its path relative to the search
% root (minus the folder itself) is used as the "mouse" label.
    % Every file name ends in ddd_ddd. Position cfg.RidPosition (1 or 2) of
    % that pair is the recording id (rid) matching imaging to behavior; the
    % other position is the segment index. The reference is the .sbx whose
    % ddd_ddd matches cfg.refid ('d' = any digit, e.g. 'ddd_000').
    % behavior files: <anything>ddd.mat containing a behavTag (if given),
    % excluding the Scanbox info .mat of every .sbx in the folder
    % <anything>_<rid ddd>_<yyyy.mm.dd>_<HH.MM.SS>.mat, 
    % tokens: 1 = rid, 2 = date
%%%%%%%%%%%%%%%%%%
%%% USER INPUT %%%
%%%%%%%%%%%%%%%%%%
clear
cfg.rootDir   = 'Y:\';

% empty to find all directories
cfg.mouseList = {'mouse1'};   % {} = search all of rootDir; or e.g. {'ADles2'}
cfg.dateList  = {'260930'};   % {} = every folder under each mouse (any depth) that directly

% folders to add to the path
cfg.codePaths = {
    'C:\Users\sammxie\Documents\Github_personal\DataProcessing'  % split_and_bin etc.
    'C:\Users\sammxie\OneDrive - Boston University\Howelab\Code\Calcium_2p_preprocessing\' 
    };

% which steps to run
cfg.steps.exportTifs     = false;  % 0b. optional: per-channel raw tifs of every .sbx
                                   %     Slow and repetitive file, only for inspection
cfg.steps.reference      = true;   % 1.  motion-correct rid 000 (red channel)
cfg.steps.splitBehavior  = true;   % 2a. split & bin behavior
cfg.steps.motionCorrect  = true;   % 2c. motion-correct each session
cfg.steps.behaviorConcat = true;   % 2d. concatenated, aligned behavior
cfg.steps.registerRef    = true;   % 2e. register reference to session
cfg.overwrite = false;             % true = redo steps whose outputs already exist

% recordings
cfg.RidPosition = 1; % rrr_ddd; rrr = recording ID (diff FOV), ddd = segmented movies 
cfg.refid      = 'ddd_000';   % search key for refernece move
cfg.refChannel  = 2;       % PMT channel of the reference (2 = red)
cfg.funcChannel = 1;       % PMT channel of the functional movie (1 = green)
cfg.behavTag    = {};% Find behavior based on keyword
cfg.recordingTag = {'acq'}; % Filter sbx based this 

% behavior split & bin
cfg.ttlField  = 'ttlIn2';  % TTL field Scanbox frames arrive on (movie file names use it)
cfg.splitArgs = {};        % extra inputs to split_and_bin_grid_behav_files, e.g. {'virmenStartTTL', 3}
cfg.maxFrameMismatch = 5; % max |behavior frames - sbx frames| per segment

% motion correction options passed to sbx_motion_correct (see its help).
% Do not set 'outfile', 'channel', 'dropFirstFrame' or 'crop' here.
cfg.mcOpts = {'maxShift', 0.1, 'nonrigid', true};

% raw-frame crop [row1 col1 row2 col2] used everywhere (motion correction
% and raw tif export). Columns 1-79 are the bidirectional scan turnaround
% band (saturated, alternating rows); Inf = to the frame edge.
cfg.crop = [1 80 Inf Inf];

%%%%%%%%%%%%%%%%%%%%%%%
%%% END: USER INPUT %%%
%%%%%%%%%%%%%%%%%%%%%%%

%%
%%%%%%%%%%%%%
%%% SETUP %%%
%%%%%%%%%%%%%
close all
found = isfolder(cfg.codePaths);
cellfun(@(x) addpath(genpath(x)), cfg.codePaths(found));
if ~all(found)
    warning('Code path(s) not found: %s', strjoin(cfg.codePaths(~found), ', '));
end
required = {'sbx_motion_correct', 'sbx_file_info', 'split_and_bin_grid_behav_files', ...
    'check_TTL_spacing', 'vec'};
missing = required(cellfun(@(f) exist(f, 'file') == 0, required));
if ~isempty(missing)
    error(['Functions not on the path: %s. (vec.m comes with CVX / SeDuMi, e.g. ' ...
        'ImageProcessing\\utils\\cvx-w64.)'], strjoin(missing, ', '));
end
reserved = {'outfile', 'channel', 'dropFirstFrame', 'crop'};
if any(ismember(lower(cfg.mcOpts(1:2:end)), lower(reserved)))
    error('mcOpts must not set %s; the batch script sets them.', strjoin(reserved, ', '));
end
warning('off', 'MATLAB:polyshape:repairedBySimplify')

diary(fullfile(cfg.rootDir, ['batch_2p_Calcium_log_' datestr(now, 'yyyymmdd_HHMMSS') '.txt']))
fprintf('batch_2p_Calcium started %s\nrootDir: %s\n', datestr(now), cfg.rootDir);

if isempty(gcp('nocreate')), parpool('Threads'); end

% Get the files to be processed
summary = cell(0, 4);   % {mouse, date, session, status}
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
else % recursive search .sbx
    searchRoots = {cfg.rootDir};
    if ~isempty(cfg.mouseList)
        searchRoots = fullfile(cfg.rootDir, cfg.mouseList);
    end
    for r = 1:numel(searchRoots)
        dd = dir(fullfile(searchRoots{r}, '**', '*.sbx'));
        found = sort(unique({dd(~[dd.isdir]).folder}));
        fprintf('%s: %d folder(s) with .sbx files\n', searchRoots{r}, numel(found));
        dateDirs = [dateDirs, found]; %#ok<AGROW>
    end
    dateDirs = unique(dateDirs, 'stable');
    if ~isempty(cfg.dateList)   % dateList without mouseList: keep folders with those names
        [~, leaf] = cellfun(@fileparts, dateDirs, 'UniformOutput', false);
        dateDirs = dateDirs(ismember(leaf, cfg.dateList));
    end
    % Save date folders processed for current run
    save(fullfile(cfg.rootDir, 'dirs_w_rec.mat'), 'dateDirs');
end
if isempty(dateDirs)
    warning('No folders with .sbx files found under %s.', cfg.rootDir);
end


%%%%%%%%%%%%%%%%%
%%% MAIN LOOP %%%
%%%%%%%%%%%%%%%%%
for d = 1:numel(dateDirs)
    [parent, dateStr] = fileparts(dateDirs{d});
    label = strip(erase(parent, strip(cfg.rootDir, 'right', filesep)), 'left', filesep);

    % Steps 0-2 for one date folder. Returns {session, status} rows for the
    % summary (the reference only appears if it failed)
    fprintf('\n====================%s: %s ====================\n', label, dateStr);
    rows = cell(0, 2);

    [status, recs] = attempt(@() findRecordings(dateDirs{d}, cfg));
    if ~strcmp(status, 'ok')
        rows = {'-', status};
        return
    end
    % Step 0 (optional sbx2tif conversion)
    if cfg.steps.exportTifs
        allSbx = [recs.ref, recs.sessions.sbx];
        for k = 1:numel(allSbx)
            status = attempt(@() sbx2tif_samme(allSbx{k}, 'crop', cfg.crop, ...
                'outDir', fullfile(dateDirs{d}, 'raw_tifs'), 'overwrite', cfg.overwrite));
            if ~strcmp(status, 'ok')
                rows(end+1, :) = {['tif ' nameOf(allSbx{k})], status}; 
            end
        end
    end
    % Step 1: motion correct reference
    if cfg.steps.reference
        status = attempt(@() processReference(dateDirs{d}, recs, cfg));
        if ~strcmp(status, 'ok')
            rows(end+1, :) = {'reference', status};
        end
    end

    file_locs = struct([]);
    matchedBeh_Tif = cell(0, 4);
    for s = 1:numel(recs.sessions)
        sess = recs.sessions(s);
        fprintf('\n--- session %s (rid %s, %d segment(s)) ---\n', sess.name, sess.rid, numel(sess.sbx));
        [status, fl, matchRows] = attempt(@() processSession(dateDirs{d}, sess, cfg));
        rows(end+1, :) = {sess.name, status}; 
        if strcmp(status, 'ok')
            file_locs = [file_locs fl]; %#ok<AGROW>
            matchedBeh_Tif = [matchedBeh_Tif; matchRows]; %#ok<AGROW>
        end
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
diary off



%% Local functions

function recs = findRecordings(dateDir, cfg)
    % Group .sbx files by rid and pair each rid with its behavior file.
    % recs.ref      - cell of .sbx paths (no extension) of the reference rid
    % recs.sessions - struct array: rid, name, sbx (cell of paths, no
    %                 extension, sorted by segment), behavFile (full path)
    
    ridPos = cfg.RidPosition;
    segPos = 3 - ridPos;
    refRegex = ['^' strrep(cfg.refid, 'd', '\d') '$'];

    % .sbx files: <anything>ddd_ddd.sbx, containing a recordingTag (if given)
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
    if isempty(names)
        error('No .sbx files ending in ddd_ddd (with keyword %s) in %s', ...
            strjoin(cellstr(cfg.recordingTag), '/'), dateDir);
    end
    [~, order] = sortrows(str2double(tok(:, [ridPos segPos])));   % by rid, then segment
    tok = tok(order, :);
    rids = tok(:, ridPos)';
    bases = fullfile(dateDir, regexprep(names(order), '\.sbx$', ''));
    bases = reshape(bases, 1, []);
    isRef = ~cellfun(@isempty, regexp(strcat(tok(:, 1), '_', tok(:, 2))', refRegex, 'once'));

    % find behavior raw output
    [mnames, btok] = listMatching(dateDir, '*.mat', ...
        '_(\d{3})_(\d{4}\.\d{2}\.\d{2})_\d{2}\.\d{2}\.\d{2}\.mat$', 2);
    isBeh = hasKey(mnames, cfg.behavTag) & ...
        ~ismember(regexprep(mnames, '\.mat$', ''), regexprep(allSbx, '\.sbx$', ''));
    mnames = mnames(isBeh);
    btok = btok(isBeh, :);

    % references: one per rid (e.g. 001_000 for rid 001). Output name is
    % ref_field_MC.tif if the folder has one reference, else ref_field_MC_rec<rid>.tif
    recs.ref = bases(isRef);
    refRids = rids(isRef);
    [refRids, iu] = unique(refRids, 'stable');
    if numel(iu) < numel(recs.ref)
        fprintf('  ! several reference files share a rid; using the first of each\n');
    end
    recs.ref = recs.ref(iu);
    recs.refTif = repmat({'ref_field_MC.tif'}, size(recs.ref));
    if numel(recs.ref) > 1
        recs.refTif = strcat('ref_field_MC_rec', refRids, '.tif');
    end
    recs.sessions = struct('rid', {}, 'name', {}, 'sbx', {}, 'behavFile', {}, 'refTif', {});
    for rid = unique(rids(~isRef))
        rid = rid{1}; %#ok<FXSET>
        cand = find(strcmp(btok(:, 1), rid));        % behavior rid is token 1
        if numel(cand) > 1
            [~, i] = min(cellfun(@numel, mnames(cand)));     % shortest name
            fprintf('  ! rid %s: behavior files %s; using %s\n', rid, ...
                strjoin(mnames(cand), ', '), mnames{cand(i)});
            cand = cand(i);
        end
        if isempty(cand)
            fprintf('  ! rid %s: no behavior .mat with this rid (keyword %s), skipping\n', ...
                rid, strjoin(cellstr(cfg.behavTag), '/'));
            continue
        end
        % session folder: behavior name without its ddd_ddd, + _rec<rid>
        segs = strsplit(mnames{cand}, '_');
        stem = strjoin(segs(1:find(strcmp(segs, rid), 1) - 1), '_');
        % reference: same rid; else the only one; else the nearest rid
        refTif = '';
        k = find(strcmp(refRids, rid));
        if isempty(k) && numel(refRids) == 1
            k = 1;
            fprintf('  o rid %s: no own reference; using the only one (rid %s) as template\n', rid, refRids{1});
        elseif isempty(k) && ~isempty(refRids)
            [~, k] = min(abs(str2double(refRids) - str2double(rid)));
            fprintf('  ! rid %s: no own reference; using the nearest (rid %s)\n', rid, refRids{k});
        end
        if ~isempty(k)
            refTif = recs.refTif{k};
        end
        recs.sessions(end+1) = struct('rid', rid, ...
            'name', [stem '_rec' rid], ...
            'sbx', {bases(strcmp(rids, rid) & ~isRef)}, ...
            'behavFile', fullfile(dateDir, mnames{cand}), ...
            'refTif', refTif);
    end
    fprintf('  found %d session(s), %d reference file(s)\n', numel(recs.sessions), numel(recs.ref));
end

function processReference(dateDir, recs, cfg)
    % Step 1: motion-correct every reference recording (red channel).
    if isempty(recs.ref)
        fprintf('  ! no reference recording (refid %s); step 2e will be skipped\n', cfg.refid);
        return
    end
    for k = 1:numel(recs.ref)
        refOut = fullfile(dateDir, recs.refTif{k});
        if shouldRun(true, {refOut, shiftsFile(refOut)}, cfg, ['reference motion correction ' nameOf(recs.ref{k})])
            sbx_motion_correct(recs.ref{k}, 'outfile', refOut, 'channel', cfg.refChannel, ...
                'dropFirstFrame', true, 'crop', cfg.crop, cfg.mcOpts{:});
        end
    end
end

function [fl, matchRows] = processSession(dateDir, sess, cfg)
    % Steps 2a-2e for one session. Returns its file_locs entry and its rows of
    % matchedBeh_Tif.
    sessDir = fullfile(dateDir, sess.name);
    if ~isfolder(sessDir)
        mkdir(sessDir);
    end
    mcPath = fullfile(sessDir, 'ConcatenatedMovie_MC.tif');
    catPath = fullfile(sessDir, 'output_behaviorConcat.mat');
    regrefPath = fullfile(sessDir, 'ref_field_MC_registeration.tif');
    nSeg = numel(sess.sbx);

    % ---- 2a. split & bin behavior ----
    movies = findMovieFiles(sess.behavFile, cfg.ttlField);
    if shouldRun(cfg.steps.splitBehavior, numel(movies) == nSeg, cfg, 'split & bin behavior')
        [~, retry] = split_and_bin_grid_behav_files(sess.behavFile, cfg.splitArgs{:});
        if any(retry)
            fprintf('  ! split_and_bin flagged a likely virmen misalignment; check ''virmenStartTTL'' in splitArgs\n');
        end
        movies = findMovieFiles(sess.behavFile, cfg.ttlField);
    end

    % ---- 2b. match behavior movies to sbx segments ----
    nSbx = cellfun(@(f) getfield(sbx_file_info(f), 'nFramesDisk'), sess.sbx); 
    nBeh = cellfun(@(f) numel(getfield(load(f, 'timestamp'), 'timestamp')), movies); 
    fprintf('  segment  sbx frames  behavior frames\n');
    for k = 1:max(nSeg, numel(movies))
        fprintf('  %7d  %10s  %15s\n', k, numOrDash(nSbx, k), numOrDash(nBeh, k));
    end
    % match split movie with sbx
    if numel(movies) ~= nSeg
        warning('%d behavior movie(s) (TTL bouts) but %d sbx segment(s); matching existing movies...', numel(movies), nSeg);
        [iM, iS] = matchMoviesToSbx(nBeh, nSbx);
        fprintf('  matched pairs (movie -> sbx segment):\n');
        for k = 1:numel(iM)
            fprintf('    movie%d (%d fr) -> %s (%d fr)\n', iM(k), nBeh(iM(k)), nameOf(sess.sbx{iS(k)}), nSbx(iS(k)));
        end
        if nSeg > numel(iS)
            fprintf('  ! sbx without behavior, left out of this session: %s\n', ...
                strjoin(nameOf(sess.sbx(setdiff(1:nSeg, iS))), ', '));
        end
        if numel(movies) > numel(iM)
            fprintf('  ! behavior movies without sbx, ignored: %s\n', ...
                strjoin(nameOf(movies(setdiff(1:numel(movies), iM))), ', '));
        end
        movies = movies(iM);
        nBeh = nBeh(iM);
        sess.sbx = sess.sbx(iS);     % motion correction then uses only matched segments
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

    % ---- 2c. motion correction ----
    if shouldRun(cfg.steps.motionCorrect, {mcPath, shiftsFile(mcPath)}, cfg, 'motion correction')
        sbx_motion_correct(sess.sbx, 'outfile', mcPath, 'channel', cfg.funcChannel, 'dropFirstFrame', true, 'crop', cfg.crop, cfg.mcOpts{:});
    end

    % ---- 2d. concatenated behavior aligned to the movie ----
    nUsed = nSbx - 1;   % the first frame of every segment is dropped
    if shouldRun(cfg.steps.behaviorConcat, {catPath}, cfg, 'concatenate behavior')
        if isfile(shiftsFile(mcPath))
            nMovie = size(getfield(load(shiftsFile(mcPath), 'frameIndex'), 'frameIndex'), 1); 
            if nMovie ~= sum(nUsed)
                error('Motion-corrected movie has %d frames but the sbx segments give %d.', nMovie, sum(nUsed));
            end
        end
        concatenatedBehavior = concatBehavior(movies, nUsed, match); 
        save(catPath, 'concatenatedBehavior');
    end

    % ---- 2e. register reference to the session ----
    if shouldRun(cfg.steps.registerRef, {regrefPath}, cfg, 'register reference')
        registerReference(dateDir, sess.refTif, sessDir, mcPath, regrefPath);
    end

    % ---- file locations (same fields as batch_2p_processData_Lily, plus new) ----
    if ~isfile(regrefPath)
        regrefPath = '';
    end
    fl = struct( ...
        'rec_dname', sess.name, ...
        'ref_fname', sess.refTif, ...                                           % '' if none
        'behav_fname', [nameOf(sess.behavFile) '.mat'], ...
        'rec_fnames', {sbxNames}, ...                                          % now .sbx, not .tif
        'snb_behavs', {strcat(nameOf(movies), '.mat')}, ...
        'mc_path', fullfile(sess.name, 'ConcatenatedMovie_MC.tif'), ...         % relative
        'catbehav_path', fullfile(sess.name, 'output_behaviorConcat.mat'), ...  % relative
        'regref_path', regrefPath, ...                                          % absolute
        'mc_shifts_path', fullfile(sess.name, 'ConcatenatedMovie_MC_shifts.mat'), ...
        'rid', sess.rid, ...
        'match', match);
end

function cb = concatBehavior(movies, nUsed, match)
    % Stack per-frame behavior across segments, aligned to the imaging frames
    % kept by sbx_motion_correct (first frame of each segment dropped).
    nSeg = numel(movies);
    segs = cellfun(@load, movies, 'UniformOutput', false);
    isPerFrame = @(B, f) (isnumeric(B.(f)) || islogical(B.(f))) && isvector(B.(f)) ...
        && numel(B.(f)) == numel(B.timestamp);
    pf = cellfun(@(B) fieldnames(B)', segs, 'UniformOutput', false);
    pf = cellfun(@(B, fn) fn(cellfun(@(f) isPerFrame(B, f), fn)), segs, pf, 'UniformOutput', false);
    allPF = unique([pf{:}], 'stable');

    % fields that are not per-frame: keep the first segment's copy
    cb = struct();
    for f = setdiff(fieldnames(segs{1})', [allPF, {'check_ttl_spacing'}], 'stable')
        cb.(f{1}) = segs{1}.(f{1});
    end
    % per-frame fields: drop first frame, trim / pad (last value) to nUsed, stack
    for f = allPF
        parts = cell(nSeg, 1);
        for s = 1:nSeg
            if ismember(f{1}, pf{s})
                v = double(segs{s}.(f{1})(2:end));
            else
                v = [];                                   % field missing in this segment
            end
            if isempty(v)
                parts{s} = nan(nUsed(s), 1);
            else
                parts{s} = v(min(1:nUsed(s), numel(v)));  % indices past the end repeat the last value
                parts{s} = parts{s}(:);
            end
        end
        cb.(f{1}) = vertcat(parts{:});
    end

    % bookkeeping
    cb.segment = reshape(repelem((1:nSeg)', nUsed(:)), [], 1);   % column even for 1 segment
    cb.frameInSbx = cell2mat(arrayfun(@(n) (1:n)', nUsed(:), 'UniformOutput', false));
    ttlCheck = cellfun(@(B) fieldOr(B, 'check_ttl_spacing', []), segs, 'UniformOutput', false);
    [match.check_ttl_spacing] = ttlCheck{:};
    cb.segmentInfo = match;
    % first and last virmen trial of each segment are incomplete: ignore them
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

function registerReference(dateDir, refTif, sessDir, mcPath, regrefPath)
    % Step 2e: rigidly register the session's reference mean image to the
    % session mean image (multimodal: different channel and brightness).
    if isempty(refTif)
        fprintf('  ! no reference for this session; skipping registration\n');
        return
    end
    refShifts = shiftsFile(fullfile(dateDir, refTif));
    if ~isfile(refShifts) || ~isfile(shiftsFile(mcPath))
        fprintf('  ! reference or session motion correction missing; skipping registration\n');
        return
    end
    R = load(refShifts, 'meanImg');
    S = load(shiftsFile(mcPath), 'meanImg', 'maxImg');
    avg_refvid = double(R.meanImg);
    avg_mcvid = double(S.meanImg);
    max_mcvid = double(S.maxImg);
    [optimizer, metric] = imregconfig('multimodal');
    registered = imregister(avg_refvid, avg_mcvid, 'rigid', optimizer, metric);

    writeTif16(regrefPath, registered);
    writeTif16(fullfile(sessDir, 'ConcatenatedMovie_MC_avg_registered.tif'), avg_mcvid);
    writeTif16(fullfile(sessDir, 'ConcatenatedMovie_MC_max_registered.tif'), max_mcvid);
    save(fullfile(sessDir, 'all_registeration_imgs.mat'), 'registered', 'avg_mcvid', 'avg_refvid', 'max_mcvid');
    merged = imfuse(avg_mcvid, registered, 'falsecolor', 'Scaling', 'joint', 'ColorChannels', [2 1 0]);
    imwrite(merged, fullfile(sessDir, 'merged_channel.tif'));   % green = session, red = reference
end


%%%%%%%%%%%%%%%
%%% HELPERS %%%
%%%%%%%%%%%%%%%

function varargout = attempt(fn)
    % Run fn, returning 'ok' or 'FAILED: <message>' first, then fn's outputs
    % ([] on failure). Errors are printed with their stack, not rethrown.
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
    % True if the step is on and either overwrite is set or it is not done.
    % done: logical, or a cell of output files that must all exist.
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

function [names, tok, other] = listMatching(folder, pattern, regex, nTok)
    % Files in folder matching the dir pattern whose names fully match regex.
    % tok is [nFiles x nTok] cell of the regex tokens; other = non-matching names.
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

function [iM, iS] = matchMoviesToSbx(nBeh, nSbx)
    % Pair behavior movies with sbx segments when their counts differ.
    % Rules: (1) order is kept: movie1,2,3... pair with sbx segments in
    % increasing order (no crossing); (2) among such pairings, all of the
    % shorter list is paired and the total |frame difference| is minimal.
    % Dynamic programming: D(i+1,j+1) = best cost pairing the first i items
    % of the shorter list within the first j items of the longer one.
    % Returns index vectors: movie iM(k) <-> sbx segment iS(k).
    swap = numel(nBeh) > numel(nSbx);   % true: sbx list is the shorter one
    if swap
        a = nSbx; b = nBeh;
    else
        a = nBeh; b = nSbx;
    end
    K = numel(a);
    L = numel(b);
    D = inf(K + 1, L + 1);
    D(1, :) = 0;
    for i = 1:K
        for j = i:L
            % skip b(j), or pair a(i) with b(j)
            D(i+1, j+1) = min(D(i+1, j), D(i, j) + abs(a(i) - b(j)));
        end
    end
    % backtrack from the end to recover the pairs
    ib = zeros(1, K);
    i = K;
    j = L;
    while i > 0
        if j > i && D(i+1, j+1) == D(i+1, j)
            j = j - 1;                  % b(j) left unpaired
        else
            ib(i) = j;                  % a(i) paired with b(j)
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

function tf = hasKey(names, keys)
    % True for names containing any of the keywords (case-insensitive);
    % all true when keys is empty.
    tf = true(size(names));
    if ~isempty(keys) && ~isempty(names)
        tf = contains(names, cellstr(keys), 'IgnoreCase', true);
    end
end

function movies = findMovieFiles(behavFile, ttlField)
    % Split behavior files <behav>_<ttlField>_movie<#>.mat, sorted by number.
    [fd, fb] = fileparts(behavFile);
    stem = [fb '_' ttlField '_movie'];
    [names, tok] = listMatching(fd, [stem '*.mat'], ['^' regexptranslate('escape', stem) '(\d+)\.mat$'], 1);
    movies = {};
    if isempty(names)
        return
    end
    [~, order] = sort(str2double(tok));
    movies = reshape(fullfile(fd, names(order)), 1, []);
end

function writeTif16(path, img)
    % Single-page uint16 tif.
    t = Tiff(path, 'w');
    c = onCleanup(@() t.close()); 
    t.setTag(struct('ImageLength', size(img, 1), 'ImageWidth', size(img, 2), ...
        'Photometric', Tiff.Photometric.MinIsBlack, 'BitsPerSample', 16, ...
        'SamplesPerPixel', 1, 'SampleFormat', Tiff.SampleFormat.UInt, ...
        'PlanarConfiguration', Tiff.PlanarConfiguration.Chunky, ...
        'Compression', Tiff.Compression.None));
    t.write(uint16(img));
end

function p = shiftsFile(tifPath)
    % *_shifts.mat written next to a tif by sbx_motion_correct.
    [d, n] = fileparts(tifPath);
    p = fullfile(d, [n '_shifts.mat']);
end

function n = nameOf(paths)
    % File name without folder or extension; works on a char or a cell.
    [~, n] = cellfun(@fileparts, cellstr(paths), 'UniformOutput', false);
    if ischar(paths)
        n = n{1};
    end
end

function v = fieldOr(s, f, default)
    v = default;
    if isfield(s, f)
        v = s.(f);
    end
end

function s = numOrDash(v, k)
    s = '-';
    if k <= numel(v)
        s = num2str(v(k));
    end
end
