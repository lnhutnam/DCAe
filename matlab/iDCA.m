function [output ] = iDCA( D, lam, theta, para )
output.method = 'iDCA';

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end

objstep = 1;

maxIter = para.maxIter;
tol = para.tol*objstep;

regType = para.regType;

[row, col, data] = find(D);

[m, n] = size(D);

R = para.R;
U0 = para.U0;
U1 = U0;

V0 = para.V0';
V1 = V0;

spa = sparse(row, col, data, m, n); % data input == D

c_1 = 3;
c_2 = norm(data);

clear D;

obj = zeros(maxIter+1, 1);
obj_y = zeros(maxIter+1, 1);
RMSE = zeros(maxIter+1, 1);
trainRMSE = zeros(maxIter+1, 1);
Time = zeros(maxIter+1, 1);
Lls = zeros(maxIter+1, 1);
Ils = zeros(maxIter+1, 1);
nnzUV = zeros(maxIter+1, 2);
no_acceleration = zeros(maxIter+1, 1);

part0 = partXY(U0', V0, row, col, length(data));


ga = theta;
fun_num = para.fun_num;
% fun_num = 1;
part0 = data - part0';
obj(1) = obj_func(part0, U0, V0, lam, fun_num, ga);
obj_y(1) = obj(1);
c = 1;

L = 1;
sigma = 1e-5;

maxinneriter = 1;

Lls(1) = L;
 % testing performance
if(isfield(para, 'test'))
    tempS = eye(size(U1,2), size(V1',2));
    if(para.test.m ~= m)
        RMSE(1) = MatCompRMSE(V1', U1, tempS, para.test.row, para.test.col, para.test.data);
        trainRMSE(1) = sqrt(sum(part0.^2)/length(data));
    else
        RMSE(1) = MatCompRMSE(U1, V1', tempS, para.test.row, para.test.col, para.test.data);
        trainRMSE(1) = sqrt(sum(part0.^2)/length(data));
    end
    fprintf('method: %s data: %s  RMSE %.2d RMSE %.2d \n', output.method, para.data, RMSE(1),obj(1));
end



for i = 1:maxIter
    tt = cputime;

    y_U = U1;
    y_V = V1;

    y_obj = obj(i);
    no_acceleration(i) = i;


    obj_y(i) = y_obj;
    
    delta_U = U1 - U0;
    delta_V = V1 - V0;
    
    U0 = U1;
    V0 = V1;
    
    
    setSval(spa,part0,length(part0));
    

    
    
% --------------------------

    grad_U = -spa*y_V';
    grad_V = -y_U'*spa;
    
    if(fun_num==4)
        grad_U = grad_U - lam*ga*(1-exp(-ga*abs(y_U))).*sign(y_U) - 0.9999*L*c_2*delta_U;
        grad_V = grad_V - lam*ga*(1-exp(-ga*abs(y_V))).*sign(y_V) - 0.9999*L*c_2*delta_V;
    end
    
    
% --------------------------
    
    norm_y = norm(y_U,'fro')^2 + norm(y_V,'fro')^2;
    grad_h1_U = y_U*norm_y;
    grad_h1_V = y_V*norm_y;

    grad_h2_U = y_U;
    grad_h2_V = y_V;

    grad_h_U = c_1*grad_h1_U + c_2*grad_h2_U;
    grad_h_V = c_1*grad_h1_V + c_2*grad_h2_V;
    
    obj_h_y = c_1*0.25*(norm_y^2) + c_2*0.5*norm_y;
    
    for inneriter = 1:maxinneriter
      
        % update U, V 
        [U1, V1] = make_update_iDCAe(grad_U,grad_V,grad_h_U,grad_h_V,c_1,c_2,L,lam,ga, 6);
        
        norm_x = norm(U1,'fro')^2 + norm(V1,'fro')^2;
        
        delta_U = U1 - y_U;
        delta_V = V1 - y_V;
        
        obj_h_x = c_1*0.25*(norm_x^2) + c_2*0.5*norm_x;
        
        reg = obj_h_x - obj_h_y - sum(sum(delta_U.*grad_h_U)) - sum(sum(delta_V.*grad_h_V));
        
        part0 = sparse_inp(U1', V1, row, col);
        
        part0 = data - part0';
        
        x_obj = obj_func(part0, U1, V1, lam, fun_num, ga);
        

        
    end
    
    Lls(i+1) = L;
    Ils(i+1) = inneriter;
    
  
    % ----------------------
    c = c + 1;

    if(i > 1)
        delta = (obj(i)- x_obj)/x_obj;
    else
        delta = inf;
    end
    
    Time(i+1) = cputime - tt;
    obj(i+1) = x_obj;
    
    fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.1f; L %d; time: %.3d;  nnz U:%0.3d; nnz V %0.3d \n', ...
        i, x_obj, delta, para.maxR, lam, L, Time(i+1), nnz(U1)/(size(U1,1)*size(U1,2)),nnz(V1)/(size(V1,1)*size(V1,2)));
    
    nnzUV(i+1,1) = nnz(U1)/(size(U1,1)*size(U1,2));
    nnzUV(i+1,2) = nnz(V1)/(size(V1,1)*size(V1,2));
    
    % testing performance
    if(isfield(para, 'test'))
        tempS = eye(size(U1,2), size(V1',2));
        if(para.test.m ~= m)
            RMSE(i+1) = MatCompRMSE(V1', U1, tempS, para.test.row, para.test.col, para.test.data);
            trainRMSE(i+1) = sqrt(sum(part0.^2)/length(data));
        else
            RMSE(i+1) = MatCompRMSE(U1, V1', tempS, para.test.row, para.test.col, para.test.data);
            trainRMSE(i+1) = sqrt(sum(part0.^2)/length(data));
        end
        fprintf('method: %s data: %s  RMSE %.2d \n', output.method, para.data, RMSE(i));
    end
    
    if(i > 1 && abs(delta) < tol)
        break;
    end
    
    if(sum(Time) > para.maxtime)
        break;
    end
end

output.obj = obj(1:(i+1));
output.Rank = para.maxR;
output.RMSE = RMSE(1:(i+1));
output.trainRMSE = trainRMSE(1:(i+1));

Time = cumsum(Time);
output.Time = Time(1:(i+1));
output.U = U1;
output.V = V1;
output.data = para.data;
output.L = Lls(1:(i+1));
output.Ils = Ils(1:(i+1));
output.nnzUV = nnzUV(1:(i+1),:);
output.no_acceleration = no_acceleration(1:(i+1));
output.lambda = lam;
output.theta = ga;
output.reg = para.reg;


end


function[pk] = HS(U,lam)

    tpk = U(:);
    [~,ind] = sort(tpk,'ascend');
    tpk(ind(1:(end-lam))) = 0;
    pk = reshape(tpk,size(U,1),size(U,2));

end

