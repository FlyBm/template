# 数学

[TOC]

## 模板

### 矩阵快速幂

```cpp
typedef long long ll;

const int maxn = 2e5 + 30;
const int mod = 1e9 + 7;

struct M {
    array<ll, 3>a[3];
    explicit M() {
        for (int i = 0; i < 3; i++) a[i].fill(0);
    }

    static M normal() {
        M(m);
        for (int i = 0; i < 3; i++) {
            m.a[i][i] = 1;
        }
        return m;
    }
    M operator * (const M b) {
        M(m);
        for (int i = 0; i < 3; i++ ) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    m.a[i][j] += a[i][k] * b.a[k][j];
                    m.a[i][j] %= mod;
                }
            }
        }
        return m;
    }
    M operator ^(int k) {
        M res = normal();
        M t = *this;
        while (k) {
            if (k & 1) res = res * t;
            k >>= 1;
            t = t * t;
        }
        return res;
    }
};

array<ll, 3> operator * (M a, const array<ll, 3>  b) {
    array<ll, 3> m{};
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            m[j] += a.a[j][k] * b[k];
            m[j] %= mod;
        }
    }
    return m;
}
int A, B, C, D, P, n;

void solve(){
    A = gn(), B = gn(), C = gn(), D = gn(), P = gn(), n = gn();
    if (n == 1) {
        cout << A << '\n';
        return;
    }
    if (n == 2) {
        cout << B << '\n';
        return;
    }
    array<ll, 3> a = {B, A, P / 3};
    M(m);
    m.a[0] = {D, C, 1};
        m.a[1] = {1, 0, 0};
    m.a[2] = {0, 0, 1};

    int l, r;
    for(l = 3, r; l <= P; l = r + 1) {
        r = P / (P / l);
        a[2] = P / l;
        if (r >= n) {
            a = (m ^ (n - l + 1)) * a;
            cout << a[0] << '\n';
            return;
        }
        a = (m ^ (r - l + 1)) * a;
    }
    if (l <= n) {
        a[2] = 0;
        a = (m ^ (n - l + 1)) * a;
        cout << a[0] << '\n';
        return;
    }


}

int main() {
    int T = gn();
    while (T--) solve();
}

```

### 整除分块模板

```cpp
for(int l = 1, r; l <= n; l = r + 1) {
    r = n / (n / l);
    sum += (r - l + 1) * (n / l);
}
```

### 线性筛素数

```cpp
constexpr int N = 1e5 + 100;

int prime[N], vis[N];
int tot = 0;

void Euler() {
    vis[1] = 1;
    for(int i = 2; i < N; ++i) {
        if(!vis[i]) prime[++tot] = i;
        for(int j = 1; j <= tot; ++j) {
            if(i * prime[j] > N) break;
            vis[i * prime[j]] = 1;
            if(i % prime[j] == 0) break;
        }
    }
}
```

### 高斯消元

```cpp
struct Gauss {
    int n;

    void getn(int n) {
        this->n = n;
    }

    double gauss () {
        for(int i = 1; i <= n; ++i) {
            int pos = i;
            for(int j = i + 1; j <= n; ++j) {
                if(fabs(a[j][i]) > fabs(a[pos][i])) pos = j;
            }
            if(fabs(a[pos][i]) <= 1e-6) return 0;
            if(pos != i) swap(a[pos], a[i]);
            for(int j = i + 1; j <= n; ++j) {
                double tmp = a[j][i] / a[i][i];
                for(int k = 1; k <= n; ++k) {
                    a[j][k] -= a[i][k] * tmp;
                }
            }
        }
        double ret = 1;
        for(int i = 1; i <= n; ++i) {
            ret *= a[i][i];
        }
        return fabs(ret);
    }
};
```

### 卢卡斯定理

$C(n, m) = (C(n \% mod, m \% mod) * C(n/mod, m/mod))%mod$

```cpp
ll Lucas (ll a, ll b) {
	if(b == 0) return 1;
	ll ret = (C(a%mod, b%mod, mod)* Lucas(a/mod, b/mod))%mod;
 return ret;
}
```

### 快速幂取模

```cpp
template < typename T >
T qpow(T a, T b, T m) {
  a %= m;
  T res = 1;
  while (b > 0) {
    if (b & 1) res = res * a % m;
    a = a * a % m;
    b >>= 1;
  }
  return res;
}
```

### gcd与exgcd

```cpp
template < typename T >
T GCD(T a, T b) {
	if(b) while((a %= b) && (b %= a));
	return a + b;
}

template < typename T >
T gcd(T a, T b){
    return b == 0 ? a : gcd(b, a % b);
}

template < typename T >
void ex_gcd(T a, T b, T &x, T &y){
    if(b == 0){
        x = 1, y = 0; return;
    }
    ex_gcd(b, a % b, y, x);
    y -= (a / b) * x;
}
```

### 线性求逆元

```cpp
void init() {
	inv[1] = 1;
	for (int i = 2; i <= n; ++i) 
		inv[i] = (long long)(p - p / i) * inv[p % i] % p;
}
```

### Miller Rabin判断素数

时间复杂度$O(k \log^3n)$

```cpp
bool millerRabbin(int n) {
  if (n < 3) return n == 2;
  int a = n - 1, b = 0;
  while (a % 2 == 0) a /= 2, ++b;
  // test_time 为测试次数,建议设为不小于 8
  // 的整数以保证正确率,但也不宜过大,否则会影响效率
  for (int i = 1, j; i <= test_time; ++i) {
    int x = rand() % (n - 2) + 2, v = quickPow(x, a, n);
    if (v == 1 || v == n - 1) continue;
    for (j = 0; j < b; ++j) {
      v = (long long)v * v % n;
      if (v == n - 1) break;
    }
    if (j >= b) return 0;
  }
  return 1;
}

//****************************************************************
// Miller_Rabin 算法进行素数测试
//速度快，而且可以判断 <2^63的数
//****************************************************************
const int S=20;//随机算法判定次数，S越大，判错概率越小

//计算 (a*b)%c.   a,b都是long long的数，直接相乘可能溢出的
//  a,b,c <2^63
long long mult_mod(long long a,long long b,long long c)
{
    a%=c;
    b%=c;
    long long ret=0;
    while(b)
    {
        if(b&1){ret+=a;ret%=c;}
        a<<=1;//别手残，这里是a<<=1,不是快速幂的a=a*a;
        if(a>=c)a%=c;
        b>>=1;
    }
    return ret;
}

//计算  x^n %c
long long pow_mod(long long x,long long n,long long mod)//x^n%c
{
    if(n==1)return x%mod;
    x%=mod;
    long long tmp=x;
    long long ret=1;
    while(n)
    {
        if(n&1) ret=mult_mod(ret,tmp,mod);
        tmp=mult_mod(tmp,tmp,mod);
        n>>=1;
    }
    return ret;
}

//以a为基,n-1=x*2^t      a^(n-1)=1(mod n)  验证n是不是合数
//一定是合数返回true,不一定返回false
bool check(long long a,long long n,long long x,long long t)
{
    long long ret=pow_mod(a,x,n);
    long long last=ret;
    for(int i=1;i<=t;i++)
    {
        ret=mult_mod(ret,ret,n);
        if(ret==1&&last!=1&&last!=n-1) return true;//合数
        last=ret;
    }
    if(ret!=1) return true;
    return false;
}

// Miller_Rabin()算法素数判定
//是素数返回true.(可能是伪素数，但概率极小)
//合数返回false;

bool Miller_Rabin(long long n)
{
    if(n<2)return false;
    if(n==2)return true;
    if((n&1)==0) return false;//偶数
    long long x=n-1;
    long long t=0;
    while((x&1)==0){x>>=1;t++;}
    for(int i=0;i<S;i++)
    {
        long long a=rand()%(n-1)+1;//rand()需要stdlib.h头文件
        if(check(a,n,x,t))
            return false;//合数
    }
    return true;
}

//************************************************
//pollard_rho 算法进行质因数分解
//************************************************
long long factor[100];//质因数分解结果（刚返回时是无序的）
int tol;//质因数的个数。数组小标从0开始

long long gcd(long long a,long long b)
{
    if(a==0)return 1;//???????
    if(a<0) return gcd(-a,b);
    while(b)
    {
        long long t=a%b;
        a=b;
        b=t;
    }
    return a;
}

long long Pollard_rho(long long x,long long c)
{
    long long i=1,k=2;
    long long x0=rand()%x;
    long long y=x0;
    while(1)
    {
        i++;
        x0=(mult_mod(x0,x0,x)+c)%x;
        long long d=gcd(y-x0,x);
        if(d!=1&&d!=x) return d;
        if(y==x0) return x;
        if(i==k){y=x0;k+=k;}
    }
}
//对n进行素因子分解
void findfac(long long n)
{
    if(Miller_Rabin(n))//素数
    {
        factor[tol++]=n;
        return;
    }
    long long p=n;
    while(p>=n)p=Pollard_rho(p,rand()%(n-1)+1);
    findfac(p);
    findfac(n/p);
}

int main()
{
#ifndef ONLINE_JUDGE
    freopen("/Users/kzime/Codes/acm/acm/in", "r", stdin);
#endif
   // srand(time(NULL));//需要time.h头文件  //POJ上G++要去掉这句话
    int T;
    long long n;
    scanf("%d",&T);
    while(T--) {
        scanf("%lld",&n);
        if (n == 1) {
            cout << "no\n";
            continue;
        }
        if(Miller_Rabin(n))
        {
            cout << "no\n";
            continue;
        }
        tol=0;
        findfac(n);
        sort(factor, factor + tol);
        bool flag = 0;
        for(int i=1;i<tol;i++)
            if (factor[i] == factor[i - 1])
                flag = 1;
//        cout << '\n';
        if (flag) cout << "yes\n";
        else cout << "no\n";
//        printf("%I64d\n",ans);
        memset(factor, 0, sizeof(factor));
    }
    return 0;
}
```

### 素数 欧拉函数

```cpp
void init() {
  phi[1] = 1;
  for (int i = 2; i < MAXN; ++i) {
    if (!vis[i]) {
      phi[i] = i - 1;
      pri[cnt++] = i;
    }
    for (int j = 0; j < cnt; ++j) {
      if (1ll * i * pri[j] >= MAXN) break;
      vis[i * pri[j]] = 1;
      if (i % pri[j]) {
        phi[i * pri[j]] = phi[i] * (pri[j] - 1);
      } else {
        // i % pri[j] == 0
        // 换言之，i 之前被 pri[j] 筛过了
        // 由于 pri 里面质数是从小到大的，所以 i 乘上其他的质数的结果一定也是
        // pri[j] 的倍数 它们都被筛过了，就不需要再筛了，所以这里直接 break
        // 掉就好了
        phi[i * pri[j]] = phi[i] * pri[j];
        break;
      }
    }
  }
}
```

### 中国剩余定理

#### 算法流程[¶](https://oi-wiki.org/math/crt/#_3)

1. 计算所有模数的积 n；
2. 对于第i个方程：
   1. 计算mi = n/ni；
   2. 计算 mi 在模 ni 意义下的 逆元mi^-1 ；
   3. 计算  ci = mi*mi^-1（ **不要对 ni 取模** ）。
3. 方程组的唯一解为：$\sum_{i=1}^{k}a_ic_n(mod \ \ \ n)$  。

### 筛莫比乌斯函数

```cpp
void pre() {
  mu[1] = 1;
  for (int i = 2; i <= 1e7; ++i) {
    if (!v[i]) mu[i] = -1, p[++tot] = i;
    for (int j = 1; j <= tot && i <= 1e7 / p[j]; ++j) {
      v[i * p[j]] = 1;
      if (i % p[j] == 0) {
        mu[i * p[j]] = 0;
        break;
      }
      mu[i * p[j]] = -mu[i];
    }
  }
```

## 常用公式

### 常用数列和公式

前n项平方和公式: $\frac{n*(n + 1)*(2n + 1)}{6}$

前n项立方和公式：$\frac{n^{2}*(n+1)^{2}}{4}$

等差数列平方和 $n*a_{1}^{2}+n*(n-1)*a_{1}*d+\frac{n*(n-1)*(2*n-1)*d^{2}}{6}$

### 划分问题

$n$个点最多把直线分成$C(n,0)+C(n,1)$份
$n$条直线最多把平面分成$C(n,0)+C(n,1)+C(n,2)$份
$n$个平面最多把空间分成$C(n,0)+C(n,1)+C(n,2)+C(n,3)=\frac{n^{3}+5*n+6}{6}$份
$n$个空间最多把时空分成$C(n,0)+C(n,1)+C(n,2)+C(n,3)+C(n,4)$份

### 约瑟夫环

N个人围成一圈，从第一个开始报数，第M个将被杀掉，最后剩下一个，其余人都将被杀掉

令$f[i]$表示$i$个人玩游戏报$m$退出最后胜利者的编号 最后结果为$f[n]$ 

有以下递推式：

$\begin{equation} \left\{             \begin{array}{lr}             f[1] = 0 &          \\             f[i]=(f[i-1]+m)\%i & i > 1              \end{array} \right. \end{equation}$

### 多边形面积

点顺序给出：顺时针值为正 逆时针值为负

$S=abs(x_{1}*y_{2}-y_{1}*x_{2}+x_{2}*y_{3}-y_{2}*x_{3}+...+x_{n}*y_{1}-y_{n}*x_{1})$

### 斐波那契数列

$\begin{bmatrix} F_{n+1}&F_{n}\\F_{n}&F_{n-1}\end{bmatrix}\quad={\begin{bmatrix} 1&1\\1&0\end{bmatrix}\quad}^{n}$

$F_{n}=\frac{1}{\sqrt{5}}*[(\frac{1+\sqrt{5}}{2})^{n}- (\frac{1-\sqrt{5}}{2})^{n}]$

$gcd(f[i],f[i+1])=1$

$f[m+n]=f[m-1]*f[n]+f[m]*f[n+1]$

$gcd(f[n+m],f[n])=gcd(f[n],f[m])$

$gcd(f[n],f[n+m])=f[gcd(n,n+m)]$

如果$f[k]$能被$x$整除 则$f[k*i]$都可以被整除

$f[0]+f[1]+f[2]+...+f[n]=f[n+2]-1$

$f[1]+f[3]+f[5]+...+f[2n-1]=f[2n]$

$f[2]+f[4]+f[6]+...+f[2n]=f[2n+1]-1$

$f[0]^{2}+f[1]^{2}+f[2]^{2}+...+f[n]^{2}=f[n]*f[n+1]$

### $(a/b)\%c$

计算$(a/b)\%c$ 其中b能整除a

如果$b$与$c$互素，则$(a/b)\%c=a*b^{phi(c)-1}\%c$

如果$b$与$c$不互素，则$(a/b)\%c=(a\%bc)/b$

对于$b$与$c$互素和不互素都有$(a/b)\%c=(a\%bc)/b$成立

### 因式分解

$a^{3}\pm b^{3}=(a\pm b)(a^{2}\mp ab+b^{2})$

$a^{n}-b^{n}=\left\{               \begin{array}{lr}               (a-b)(a^{n-1}+a^{n-2}b+a^{n-3}b^{2}+...+ab^{n-2}+b^{n-1}) & n为正整数\\            (a+b)(a^{n-1}+a^{n-2}b-a^{n-3}b^{2}+...+ab^{n-2}-b^{n-1})&  n为偶数          \end{array}   \right.  $

$a^{n}+b^{n}=(a+b)(a^{n-1}-a^{n-2}b+a^{n-3}b^{2}+...-ab^{n-2}+b^{n-1})  n为奇数$

### 三角函数

$tan(\alpha \pm \beta)=\frac{tan\alpha \pm tan\beta}{1\mp tan\alpha tan\beta}$

$\frac{a}{sinA}=\frac{b}{sinB}=\frac{c}{sinC}=2R$

$a^{2}=b^{2}+c^{2}-2bc*cosA$

$S=\frac{a*b*sinC}{2}$

