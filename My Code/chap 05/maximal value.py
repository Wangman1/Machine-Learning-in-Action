"""
函数说明：梯度上升法测试函数

    求函数f(x)=-x^2 + 4x的极大值

"""
def Gradient_Ascent_test():
    def f_prime(x_old):
        return -2*x_old+4
    x_old=-1
    x_new=0
    alpha=0.01
    presision=0.00000001

    while abs(x_new-x_old)>presision:
        x_old=x_new
        x_new=x_old+alpha*f_prime(x_old)
    print(x_new)

if __name__=='__main__':
    Gradient_Ascent_test()


