
for i=1:2000
    left = ".\data\run_";
    fp = left + i;
    fp_name = fp + ".dat";
    
    crd_mat = csvread(fp_name);
    scatter3(crd_mat(:,1),crd_mat(:,2), crd_mat(:,3))
    az = 90;
    el = 0;
    %view(az, el);
    title(fp_name)
    axis([0. 1. 0. 1. 0. 1.])
    pause(.01)
end