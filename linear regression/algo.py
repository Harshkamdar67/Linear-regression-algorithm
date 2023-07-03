import numpy as np 

class LinearRegression():
    def __init__(self):
        self.coefficients=None
        self.intercepts=None
    
    def fit(self,X,y):
        X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
        
        X_transpose_X=np.dot(X.T,X)
        X_transpose_y=np.dot(X.T,y)
        
        coefficients=np.linalg.solve(X_transpose_X,X_transpose_y)
        
        self.intercepts=coefficients[0]
        self.coefficients=coefficients[1:]
    
    def predict(self,X):
        X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)     
        
        y_pred=np.dot(X,np.concatenate(([self.intercepts],self.coefficients)))  
        return y_pred 
        
        

X_train=np.array([[1],[2],[3],[4],[5]])
y_train=np.array([2,4,6,8,10])

model=LinearRegression()
model.fit(X_train,y_train)

X_test=np.array([[6],[7],[8]])
y_pred=model.predict(X_test)
print(y_pred)
