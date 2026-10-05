function [avgG, avgR, rgb] = plot_two_channel_avg(greenPath, redPath, varargin)
% PLOT_TWO_CHANNEL_AVG  Mean projections of a green and red TIFF movie + overlay.
%   plot_two_channel_avg(greenPath, redPath)
%   plot_two_channel_avg(..., 'frameSkip', 10, 'startFrame', 1, ...
%                        'register', true, 'pct', 0.001)
%
%   frameSkip  : use every Nth frame (default 1 = all frames)
%   startFrame : first frame to include (default 1)
%   register   : rigidly register red avg onto green avg (default false;
%                set true if the channels were acquired separately)
%   pct        : fraction saturated at each end for the linear stretch
%                (default 0.001 -> 0.1st / 99.9th percentiles)
%   saveDir    : output folder (default: <green movie folder>/two_channel_avg)
%   prefix     : filename prefix (default: green movie filename)
%
%   Saves <prefix>_green, <prefix>_red, <prefix>_merge, each as
%   .tif (native-resolution image), .mat (data + display limits), .eps.

ip = inputParser;
ip.addParameter('frameSkip', 1);
ip.addParameter('startFrame', 1);
ip.addParameter('register', true);
ip.addParameter('pct', 0.001);
ip.addParameter('saveDir', '');
ip.addParameter('prefix', '');
ip.parse(varargin{:});
o = ip.Results;

% accept string or char paths; [a b] on strings builds an array, not a name
greenPath = char(greenPath);
redPath   = char(redPath);
o.saveDir = char(o.saveDir);
o.prefix  = char(o.prefix);

% ---- mean projections ----
avgG = tiff_mean(greenPath, o.startFrame, o.frameSkip);
avgR = tiff_mean(redPath,   o.startFrame, o.frameSkip);

if o.register
    [optimizer, metric] = imregconfig('multimodal');
    avgR = imregister(avgR, avgG, 'rigid', optimizer, metric);
end

% ---- linear stretch (per channel) ----
[g, climG] = lin_stretch(avgG, o.pct);
[r, climR] = lin_stretch(avgR, o.pct);
rgb = cat(3, r, g, zeros(size(g)));   % R = red channel, G = green channel

% ---- plot + save (one figure per panel) ----
if isempty(o.saveDir)
    o.saveDir = fullfile(fileparts(greenPath), 'two_channel_avg');
end
if ~exist(o.saveDir, 'dir'), mkdir(o.saveDir); end
if isempty(o.prefix)
    [~, o.prefix] = fileparts(greenPath);
end
params = o;

greenMap = [zeros(256,1), linspace(0,1,256)', zeros(256,1)];
redMap   = [linspace(0,1,256)', zeros(256,2)];

% green
fig = show_panel(g, greenMap, 'Green (mean)');
save_panel(fig, ind2rgb(gray2ind(g, 256), greenMap), o.saveDir, [o.prefix '_green'], ...
    struct('avgG', avgG, 'g_stretched', g, 'climG', climG, 'params', params));

% red
fig = show_panel(r, redMap, 'Red (mean)');
save_panel(fig, ind2rgb(gray2ind(r, 256), redMap), o.saveDir, [o.prefix '_red'], ...
    struct('avgR', avgR, 'r_stretched', r, 'climR', climR, 'params', params));

% merge
fig = show_panel(rgb, [], 'Merge');
save_panel(fig, rgb, o.saveDir, [o.prefix '_merge'], ...
    struct('rgb', rgb, 'climG', climG, 'climR', climR, 'params', params));

fprintf('Saved to %s\n', o.saveDir);
end

%% helpers
function avg = tiff_mean(path, startFrame, skip)
% Running mean over frames without loading the whole movie.
warning('off','imageio:tiffmexutils:libtiffWarning');
t = Tiff(path, 'r');
c = onCleanup(@() t.close());
acc = 0; n = 0; k = 1;
while true
    if k >= startFrame && mod(k - startFrame, skip) == 0
        acc = acc + double(t.read());
        n = n + 1;
    end
    if t.lastDirectory, break; end
    t.nextDirectory();
    k = k + 1;
end
if n == 0, error('No frames read from %s', path); end
avg = acc / n;
end

function [out, clim] = lin_stretch(img, pct)
% Linear stretch to [0 1]; pct of pixels saturated at each end.
% clim returns the [lo hi] raw intensities used, for reporting.
v = sort(img(~isnan(img)));
n = numel(v);
lo = v(max(1, round(pct * n)));
hi = v(min(n, round((1 - pct) * n)));
out = min(max((img - lo) ./ (hi - lo), 0), 1);
clim = [lo hi];
end

function fig = show_panel(img, cmap, ttl)
fig = figure('Color','w');
ax = axes(fig);
if isempty(cmap)
    image(ax, img);
else
    imagesc(ax, img, [0 1]);
    colormap(ax, cmap);
end
axis(ax, 'image', 'off');
title(ax, ttl);
end

function save_panel(fig, rgbImg, saveDir, name, dat)
% .tif : pixel-exact image at native resolution (no figure resampling)
% .mat : underlying image data, display limits and parameters
% .eps : vector figure for Illustrator
base = fullfile(saveDir, name);
imwrite(im2uint8(rgbImg), [base '.tif']);
save([base '.mat'], '-struct', 'dat');
set(fig, 'Renderer', 'painters');
saveas(fig, [base '.eps'], 'epsc');
saveas(fig, [base '.fig'], 'fig');
end