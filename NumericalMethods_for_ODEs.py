import numpy as np
import matplotlib.pyplot as plt

#true solution
def y(t) :
    #return np.tan(np.log(t))*t
    return np.e**t/(np.e**t+1)

#ode function
def f(t,y) :
    f = y *(1-y)
    #f = 1+y/t+(y/t)**2
    return f

def EulerMethod(t,y,h) :
    y_1 = y+h*f(t,y)
    return y_1

def rk4(t,y, h) :
    k1 = f(t,y)
    k2 = f(t+h/2,y+h/2*k1)
    k3 = f(t+h/2,y+h/2*k2)
    k4 = f(t+h,y+h*k3)
    y_1 = y + 1/6*h*(k1 + 2*k2+2*k3+k4)
    return y_1
    
def Adams(tn_1,yn_1,tn,yn,h) :
    y = yn + 3/2*h*f(tn,yn) -1/2*h*f(tn_1,yn_1)
    return y

iteration_count = 10
h = 0.1
y_0 = 0.5
t_0 = 0

#results for RK4 method
results1 = np.zeros(iteration_count+1)
results1[0] = y_0

y_current = y_0
t_current = t_0
for i in range(iteration_count) :
    y_current = rk4(t_current,y_current,h)
    t_current += h
    results1[i+1] = y_current

#result for Euler Method
results2 = np.zeros(iteration_count+1)
results2[0] = y_0
y_current = y_0
t_current = t_0
for  i in range(iteration_count) :
    y_current = EulerMethod(t_current,y_current,h)
    t_current += h
    results2[i+1] = y_current


iteration_count = 10
t_0 = 0
h=0.1
t = np.arange(t_0,t_0+h*(iteration_count+1),h)
y_0 = 0.5
#result for Adam's Method
Adams_y = np.zeros(iteration_count+1)
Adams_y[0] = y_0

#first, we get y_1
Adams_y[1] = EulerMethod(t[0],Adams_y[0],h)
for i in range(2,Adams_y.size) :
    Adams_y[i] = Adams(t[i-2],Adams_y[i-2],t[i-1],Adams_y[i-1],h)
    
print("RK4: ", results1)
print("Euler: ", results2)
print("Adams: ", Adams_y)

t = t[0:iteration_count+1]
true_y = y(t)

#plot the values
print("True value: ", true_y)
plt.plot(t,results1, label = "RK4")
plt.plot(t,results2,label = "Euler")
plt.plot(t,Adams_y,label = "Adams")
plt.plot(t,true_y, label = "True")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.show()