import random, math
import scipy.stats

x1min = 15
x1max = 45
x2min = 30
x2max = 80
x3min = 15
x3max = 45
N = 8
neadekvat=1
tSum=[0,0,0,0,0,0,0,0]

k = 0

Average_max=(x1max+x2max+x3max)/3
Average_min=(x1min+x2min+x3min)/3

ymin=round(200+Average_min)
ymax=round(200+Average_max)

X = [[-1.0, -1.0, -1.0],
     [-1.0, -1.0, 1.0],
     [-1.0, 1.0, -1.0],
     [-1.0, 1.0, 1.0],
     [1.0, -1.0, -1.0],
     [1.0, -1.0, 1.0],
     [1.0, 1.0, -1.0],
     [1.0, 1.0, 1.0]]

MatrixX = [ [ x1min , x2min , x3min ] ,
         [ x1min , x2min , x3max ] ,
         [ x1min , x2max , x3min ] ,
         [ x1min , x2max , x3max ] ,
         [ x1max , x2min , x3min ] ,
         [ x1max , x2min , x3max ] ,
         [ x1max , x2max , x3min ] ,
         [ x1max , x2max , x3max ] ]

print("Матриця X: ")
for i in range(len(MatrixX)):
    print(MatrixX[i])
for i in range (100):
    while True:
        if neadekvat==1:
            m=3
            print("Рівняння регресії: \n y=b0+b1*x1+b2*x2+b3*x3+b12*x1*x2+b13*x1*x3+b23*x2*x3+b123*x1*x2*x3")
        neadekvat=0
        MatrixY, Average, Dispersion, Beta, t = [], [], [], [], []
        for i in range(0, 8):
            MatrixY.append([random.randint(ymin, ymax) for j in range(0, m)])
            Average.append(sum(MatrixY[i]) / len(MatrixY[i]))
            Dispersion.append(sum((k - Average[i]) ** 2 for k in MatrixY[i]) / len(MatrixY[i]))
        b0 = sum([Average[i] for i in range(len(MatrixX))]) / len(MatrixX)
        b1 = sum([X[i][0] * Average[i] for i in range(len(MatrixX))]) / len(MatrixX)
        b2 = sum([X[i][1] * Average[i] for i in range(len(MatrixX))]) / len(MatrixX)
        b3 = sum([X[i][2] * Average[i] for i in range(len(MatrixX))]) / len(MatrixX)
        b12 = sum([X[i][0] * X[i][1] * Average[i] for i in range(len(MatrixX))]) / len(MatrixX)
        b13 = sum([X[i][0] * X[i][2] * Average[i] for i in range(len(MatrixX))]) / len(MatrixX)
        b23 = sum([X[i][1] * X[i][2] * Average[i]for i in range(len(MatrixX))]) / len(MatrixX)
        b123 = sum([X[i][0] * X[i][1] * X[i][2] * Average[i] for i in range(len(MatrixX))]) / len(MatrixX)
        print('Отримане рівняння регресії: \n', round(b0, 3), ' + ', round(b1, 3), ' * x1 +', round(b2, 3),
              ' * x2 +', round(b3, 3), ' * x3 +', round(b12,3),' * x1*x2 +', round(b13,3),' * x1*x3 +',round(b23,3)
              ,' * x2*x3 +', round(b123,3),' * x1*x2*x3')
        Gp = max(Dispersion) / sum(Dispersion)
        f1 = m - 1
        f2 = N
        q = 0.05
        tableGt = {2: 7679, 3: 0.6841, 4: 0.6287, 5: 0.5892, 6: 0.5598, 7: 0.5365, 8: 0.5175, 9: 0.5017, 10: 0.4884}
        tableGt2 = [(range(11, 17), 0.4366), (range(17, 37), 0.3720), (range(37, 145), 0.3093)]
        if m<11:
            Gt= tableGt.get(m)
        else:
            for i in range(len(tableGt2)):
                if m in tableGt2[i][0]:
                    Gt = tableGt2[i][1]
                    break
        if Gp < Gt:
            pass
        else:
            m += 1
            continue
        S2betaSum = sum(Dispersion) / N
        S2beta = S2betaSum / (N * m)
        Sbeta = math.sqrt(S2beta)
        MatrixCodeX =  [[1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0],
              [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
              [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
              [1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
              [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
              [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
              [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        for i in range(N):
            Beta.append(round(sum([MatrixCodeX[j][i] * Average[j] for j in range(len(MatrixCodeX))])/N,3))
            t.append(round(abs(Beta[i]/Sbeta),3))
        print("t=", t)
        f3 = f1 * f2
        tableS = round(scipy.stats.t.ppf((1 + (1 - q)) / 2, f3),3)
        b = [b0, b1, b2, b3, b12, b13, b23, b123]

        for i in range(N):
            tSum[i]+=t[i] # Сумування кожного t кожного циклу
            k+=1 # Сумування кількості чисел t
            if t[i] < tableS:
                b[i] = 0
        y = []
        for i in range(N):
            y.append(b[0] + b[1] * X[i][0] + b[2] * X[i][1] + b[3] * X[i][2] + b[4] * X[i][0]* X[i][1] +
                 b[5] * X[i][0]* X[i][2] + b[6] * X[i][1]* X[i][2] + b[7] * X[i][0]* X[i][1]* X[i][2])
            y[i]=round((y[i]),3)

        d = 0

        for i in range(len(b)):
            if b[i] != 0:
                d += 1
        f4 = N - d
        Sum = 0
        for i in range(len(y)):
            Sum += pow((y[i] - Average[i]), 2)
        Sad = (m / (N - d)) * Sum
        Fp = Sad / S2betaSum
        Ft = round(scipy.stats.f.ppf(1 - q, f4, f3), 3)
        if Fp > Ft:
            neadekvat=1
            continue
        else:
            break
for i in range (8):
    print("Середнэ значення t",i,"=",tSum[i]/100)