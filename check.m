%% gradient check
% define
clc; clear;
lin = @mylinear;
sig = @sigmoid;
rel = @relu;
alpha = 0.4;
lamb = 0.01;
x = [1 5 3 4 3
     2 6 7 4 1
     6 2 4 9 2
     1 3 4 9 1];
y = [1 0 1 0 1
     0 1 0 1 0];
batchSize = 3;
batchCount = 2;
dataSize = 5;
L = 4;
layer = [4 6 3 2];
act = {sig, sig, lin};
w = cell(1,L-1);
for l = 1:L-1
    bound = sqrt(6.0/(layer(l)+layer(l+1)));
    w{l} = 2 * bound * rand(layer(l+1), layer(l)) - bound;
end
% simulate
for iter = 1:3
    idxs = randperm(dataSize);
    % for each batch
    for k = 1:batchCount
        a = cell(1,L);
        z = cell(1,L);
        delta = cell(1,L);

        startIdx = (k-1)*batchSize+1;
        endIdx = min(k*batchSize, dataSize);
        realSize = endIdx - startIdx + 1;

        % forward computation
        a{1} = x(:,idxs(startIdx:endIdx));
        ay = y(:,idxs(startIdx:endIdx));
        for l=1:L-1
            [a{l+1}, z{l+1}] = fc(w{l}, a{l}, act{l});
        end
        J = cost(a{L},ay,w,lamb,dataSize);
        % backward
        delta{L} = dcost(a{L},ay);
        grad_w = cell(1,L-1);
        for l=L-1:-1:2
            delta{l} = bc(w{l}, z{l}, delta{l+1}, act{l-1});
        end
        for l=1:L-1
            grad_w{l} = delta{l+1} * a{l}' + lamb * w{l};
        end

        % number gradient
        epsilon = 1e-4;
        num_grad_w = cell(1,L-1);

        for gl = 1:L-1
            num_grad_w{gl} = zeros(size(w{gl}));
            for i = 1:size(w{gl},1)
                for j = 1:size(w{gl},2)
                    wminus = w;
                    wminus{gl}(i,j) = wminus{gl}(i,j) - epsilon;
                    ma = cell(1,L);
                    ma{1} = a{1};
                    for l=1:L-1
                        [ma{l+1}, ~] = fc(wminus{l}, ma{l}, act{l});
                    end
                    Jminus = cost(ma{L},ay,wminus,lamb,batchSize);

                    wplus = w;
                    wplus{gl}(i,j)  = wplus{gl}(i,j)  + epsilon;
                    pa = cell(1,L);
                    pa{1} = a{1};
                    for l=1:L-1
                        [pa{l+1}, ~] = fc(wplus{l},  pa{l}, act{l});
                    end
                    Jplus  = cost(pa{L},ay,wplus,lamb,batchSize);

                    num_grad_w{gl}(i,j) = (Jplus - Jminus) / (2*epsilon);
                end
            end
            diff = sqrt(sum(sum(num_grad_w{gl}-grad_w{gl}).^2));
            fprintf('Layer %d, check %d\n', gl, diff<1e-5)
        end
    end
end