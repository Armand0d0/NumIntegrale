#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

double* op1(double* d,int *l){
        int n = *l;
    double* d2 = (double*)malloc(sizeof(double)*n*3);
    for(int i=0;i<n;i++){
        d2[i] = d[i];
    d2[n+i]=1/d[i];
    d2[n*2+i]=sqrt(d[i]);
    }
    *l=n*3;
    return d2;
}
double* op2(double* d,int *l){
    int n = *l;
    double* d2 = (double*)malloc(sizeof(double)*n*n*3);

    for(int i=0;i<n;i++){
        for(int k=0;k<n;k++){
            d2[i*n+k]=d[i]+d[k];
            d2[n*n+i*n+k]=d[i]*d[k];
            d2[n*n*2+i*n+k]=d[i]-d[k]; 
        } 
    }
    *l = n*n*3;
    return d2;
}
/*double* fun1(double* d,int *l){
        int n = *l;
    double* d2 = (double*)malloc(sizeof(double)*n*n/2);
    for(int i=0;i<n;i++){
        for(int j=i;j<n;j++){
            d2[i] = d[i];
            d2[n+i]=1/d[i];
            d2[n*2+i]=sqrt(d[i]);
        }
    }
    *l=n*n/2;
    return d2;
}*/
void show(double* d, int n){
        double near=0;
        double min=INFINITY;
        int index=-1;
        for(int i=0;i<n;i++){
        // cout<<d[i]<<"  ";
            double di = abs(d[i]-0.0695179000);
            if(di<min){
                min = di;
                near = d[i];
                index = i;
            }
        }

        cout<<endl<<"near : "<<near<<"  index : "<<index<<endl;
        cout<<"dist : "<<min<<endl;
}

enum funcOP{
    COMP,
    MUL,
    SUM,
};
/*struct func{
    int nOP;
    funcOP* op;
    //int nFunc = nOP+1;
    int* funcIds;
};//*/
struct func{
    funcOP opType;
    struct func* func1;
    struct func* func2;
    int funcId1;
    int funcId2;


};

double eval(struct func* fun, double x, double  (*F[])(double)){////////////////////////////////////////////////////////////////////////
    if(fun->func1 != NULL && fun->func2 != NULL){
        if(fun->opType == COMP){
           return eval(fun->func1,   eval(fun->func2,x,F)   ,F);

        }else if(fun->opType == MUL){
            return eval(fun->func1,x,F) * eval(fun->func2,x,F);
        }else if(fun->opType == SUM){
            return eval(fun->func1,x,F) + eval(fun->func2,x,F);
        }
    }else if(fun->func2 != NULL){
        if(fun->opType == COMP){
           return F[fun->funcId1](  eval(fun->func2,  x  ,F)   );

        }else if(fun->opType == MUL){
            return  F[fun->funcId1](x) * eval(fun->func2,x,F);
        }else if(fun->opType == SUM){
            return  F[fun->funcId1](x) + eval(fun->func2,x,F);
        }
    }else{
        if(fun->opType == COMP){
           return F[fun->funcId1](   F[fun->funcId2](x)   );

        }else if(fun->opType == MUL){
            return F[fun->funcId1](x) * F[fun->funcId2](x);
        }else if(fun->opType == SUM){
            return F[fun->funcId1](x) + F[fun->funcId2](x);
        }
    }
    return -1;
}
void printFunc(struct func * f){
    char c=0;
    if(f->opType==COMP){
      c = 'o';
    }else if(f->opType==MUL){
        c='*';
    }else if(f->opType==SUM){
        c='+';
    }
    if(f->func1!=NULL && f->func2!=NULL){
        cout<<"(";
        printFunc(f->func1);
        cout<<") "<<c<<" (";
        printFunc(f->func2);
        cout<<")";
    }else if(f->func2 != NULL){
        cout<<" (f"<<f->funcId1<<c<<"(";
        printFunc(f->func2);
        cout<<"))";
    }else{
        cout<<"(f"<<f->funcId1<<c<<"f"<<f->funcId2<<")";
    }
}
void show2(struct func *fun[], int n,double* b, int nb,double  (*F[])(double) ){////////
    double near=0;
    double min=INFINITY;
    struct func *fx;
    int b1 = -1,b2 = -1;
    for(int i=0;i<n;i++){
    // cout<<d[i]<<"  ";
    for(int j=0;j<nb;j++){
        for(int k=j+1;k<nb;k++){
            double ev = (eval(fun[i],b[k],F)-eval(fun[i],b[j],F));
                double di = abs( ev - 0.069517900000);
                if(di<min){
                    min = di;
                    near = ev;
                    fx = fun[i];
                    b1 = j;
                    b2 = k;
                }
                //cout<<ev<<" ";
        }
    }
    }
    cout<<endl<<"near : "<<near<<"  fonction : "; printFunc(fx);
    cout<<endl<<" bornes : "<<b1<<"  " << b2<<endl;
    cout<<"dist : "<<min<<endl;
}
double id(double f){ return f;  }
double f1(double f){ return 1;  }
double f2(double f){ return 2;  }
double f3(double f){ return 3;  }
double f4(double f){ return 4;  }
double f5(double f){ return 5;  }
double f6(double f){ return 6;  }
double f7(double f){ return 7;  }
double f8(double f){ return 8;  }
double f9(double f){ return 9;  }
double fpi(double f){ return M_PI;  }
double fe(double f){ return M_E;  }
double p2(double f){ return f*f;  }
double p3(double f){ return f*f*f;  }
double p4(double f){ return f*f*f*f;  }
double inv(double f){ return 1/f;  }

struct func* create( int layers, double  (*F[])(double)){
            struct func* Fun =  (struct func*)malloc(sizeof(struct func));
            int r = rand()%3;
            if(r==0)
                Fun->opType = COMP;
            else if(r==1)
                Fun->opType = MUL;
            else
                Fun->opType = SUM;
            
            if(layers>1){
                int r2 = rand()%2;
                if(r2){
                Fun->funcId2 = 0;
                Fun->funcId1 = 0;
                Fun->func1 = create(layers-1,F);
                Fun->func2 = create(layers-1,F);
                }else{
                Fun->func1 = NULL;
                Fun->func2 = create(layers-1,F);
                Fun->funcId1 = rand()%20 +11;
                Fun->funcId2 = 0;
                }
            }else if(layers==1){
                Fun->func1 = NULL;
                Fun->func2 = NULL;
                Fun->funcId1 = rand()%20 +11;
                Fun->funcId2 = rand()%31;
                //cout<<"  rand " << Fun->funcId1<<"  "<< Fun->funcId2<<endl;
            }
            return Fun;
}

struct func** createArr(int nf, int layers, double  (*F[])(double)){
        struct func** Func =  (struct func**)malloc(sizeof(struct func*)*nf);
        for(int i=0;i<nf;i++){
            Func[i] = create(layers,F);
        }
    return Func;
}
void destroy(struct func* f){
    if(f->func1!=NULL)
        destroy(f->func1);
    if(f->func2!=NULL)
        destroy(f->func2);
    free(f);
}
void destroyArr(struct func** f,int n){
    for(int i =0;i<n;i++){
        destroy(f[i]);
    }
}
int main(){


cout.precision(15);

double  (*F[])(double)= {id,f1,f2,f3,f4,f5,f6,f7,f8,f9,fpi,fe,p2,p3,p4,inv,sqrt,log,exp,sin,cos,tan,asin,acos,atan,sinh,cosh,tanh,acosh,asinh,atanh};
//                        0  1  2  3  4 5               10    12       15                   20                      25   26   27    28    29    30

//(f21*( (f24o( (f17*((f26of24)))))))
//[] 12  -> 20
/*near : 0.0695179882902212  fonction :  (f13*(((f30+f24)) + ((f17+f10))))
 bornes : 15  20
 */

//free(f);

double m1[] = {1,2,3,4,5,6,7,8,9,M_PI,M_E};
int l=11;
double *m2 = op1(&m1[0],&l);//1,2,3,4,5,6,7,8,9,M_PI,M_E,   1,2,3,4,5,6,7,8,9,M_PI,M_E,   1  ,2, 3 ,4,5,6,7,8,9,M_PI,M_E
//                            0 1 2              9    10    11                  20 21     22 23  24             31  32
/*for (int i=0;i<l;i++){
    cout<<i<<" : "<<m2[i]<<endl;
}*/
/*                          
struct func fun1 = {.opType = COMP,.func1 =NULL,.func2 = NULL,.funcId1 = 15, .funcId2 = 12};
struct func fun2 = {.opType = SUM,.func1 =NULL,.func2 = NULL,.funcId1 = 10, .funcId2 = 0};

struct func fun = {.opType = COMP,.func1 = &fun1,.func2 = &fun2};

cout<<"EVAL : "<<eval(&fun,15,f,31 )<<endl;
struct func *farr[] = {&fun1,&fun2,&fun};//*/
int nf = 50000;
int nl = 1;
int iter = 1;
cout<<" function nb = "<<nf<<" layer nb = "<<nl<<"iter = "<<iter<<endl;
for(int i=0;i<iter;i++){
    srand(time(NULL));
    struct func **Farr = createArr(nf,nl,F);
    show2(Farr,nf,m2,l,F);
    destroyArr(Farr,10);
}
/*double *m3 = op2(m2,&l);
double *m4 = op2(m3,&l);
double *m5 = op1(m4,&l);
//double *m6 = op1(m5,&l);
show(m5,l);//*/

//cout<<sqrt((m2[20] - m2[18])* m2[18]*m2[15]   )  <<"<<<"<<endl;//sqrt(((1/M_PI)-(1/8))/(5*8))
/*for(int i=0;i<31;i++){
       cout<<f[i](2)<<endl;

} //*/
free(m2);
//free(f1);
/*
free(m3);
free(m4);
free(m5);
//free(m6);
//*/
    return 0;
}