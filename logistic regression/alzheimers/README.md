# Parameters

The files in Parameter folder are the w and c obtained from SLEP using the following code.
```
data  = load("ad_data.mat")
X_train = data.("X_train")
y_train = data.("y_train")
X_test = data.("X_test")
y_test = data.("y_test")

opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000;

[w, c] = LogisticR(X_train, y_train, 1e-2, opts)
```
Because I'm not familiar with the MATLAB, I just store all the parameters and run with python code in disease.py file.
