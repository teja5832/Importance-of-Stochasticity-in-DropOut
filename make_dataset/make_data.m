dotsize = 12;

colormap([1 0 .5;   % magenta
           0 0 .8;   % blue
           0 .6 0;   % dark green
           .3 1 0]); % bright green

f = figure;x=clusterincluster(1000);
csvwrite_with_headers('clusterincluster.csv', x, {'x','y','label'});
scatter(x(:,1), x(:,2), dotsize, x(:,3)); axis equal;saveas(f, 'clusterincluster.png');
f = figure;y=clusterincluster(100);
csvwrite_with_headers('clusterincluster_test.csv', y, {'x','y','label'});scatter(y(:,1), y(:,2), dotsize, y(:,3)); axis equal;saveas(f, 'clusterincluster_test.png');

f = figure;x=corners(1000);
csvwrite_with_headers('corners.csv', x, {'x','y','label'});
scatter(x(:,1), x(:,2), dotsize, x(:,3)); axis equal;saveas(f, 'corners.png');
f = figure;y=corners(100);
csvwrite_with_headers('corners_test.csv', y, {'x','y','label'});scatter(y(:,1), y(:,2), dotsize, y(:,3)); axis equal;saveas(f, 'corners_test.png');

f = figure;x=crescentfullmoon(1000);
csvwrite_with_headers('crescentfullmoon.csv', x, {'x','y','label'});
scatter(x(:,1), x(:,2), dotsize, x(:,3)); axis equal;saveas(f, 'crescentfullmoon.png');
f = figure;y=crescentfullmoon(100);
csvwrite_with_headers('crescentfullmoon_test.csv', y, {'x','y','label'});scatter(y(:,1), y(:,2), dotsize, y(:,3)); axis equal;saveas(f, 'crescentfullmoon_test.png');

f = figure;x=halfkernel(1000);
csvwrite_with_headers('halfkernel.csv', x, {'x','y','label'});
scatter(x(:,1), x(:,2), dotsize, x(:,3)); axis equal;saveas(f, 'halfkernel.png');
f = figure;y=halfkernel(100);
csvwrite_with_headers('halfkernel_test.csv', y, {'x','y','label'});scatter(y(:,1), y(:,2), dotsize, y(:,3)); axis equal;saveas(f, 'halfkernel_test.png');

f = figure;x=outlier(1000);
csvwrite_with_headers('outlier.csv', x, {'x','y','label'});
scatter(x(:,1), x(:,2), dotsize, x(:,3)); axis equal;saveas(f, 'outlier.png');
f = figure;y=outlier(100);
csvwrite_with_headers('outlier_test.csv', y, {'x','y','label'});scatter(y(:,1), y(:,2), dotsize, y(:,3)); axis equal;saveas(f, 'outlier_test.png');

f = figure;x=twospirals(1000);
csvwrite_with_headers('twospirals.csv', x, {'x','y','label'});
scatter(x(:,1), x(:,2), dotsize, x(:,3)); axis equal;saveas(f, 'twospirals.png');
f = figure;y=twospirals(100);
csvwrite_with_headers('twospirals_test.csv', y, {'x','y','label'});scatter(y(:,1), y(:,2), dotsize, y(:,3)); axis equal;saveas(f, 'twospirals_test.png');

