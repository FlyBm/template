# 数学

[TOC]

## 模板

### 高斯消元求线性方程组和异或方程组

线性方程组

```cpp
#include<bits/stdc++.h>
using namespace std;
#define MAX_SIZE 1048
int Matrix[MAX_SIZE][MAX_SIZE];
int Free_x[MAX_SIZE]; //自由变元
int X_Ans[MAX_SIZE]; //解集
int Free_num=0;    //自由变元数

// 下标从0开始
int Guass(int Row,int Column) //系数矩阵的行，列
{
        int row = 0, col = 0, max_r;
        for(row = 0; row < Row && col < Column; row++, col++) {
                max_r = row;
                for(int i = row + 1;i < Row; i++)        //找出当前列的最大值
                        if(abs(Matrix[i][col]) > abs(Matrix[max_r][col]))
                                max_r = i;
                if(Matrix[max_r][col] == 0) {     //最大值为0，等价有自由元，记录
                        row--;
                        Free_x[++Free_num] = col + 1;
                        continue;
                }
                if(max_r != row)     //将最大值换到当前行
                        swap(Matrix[row], Matrix[max_r]);
                for(int i = row + 1; i < Row; ++i) {
                        if(Matrix[i][col] != 0) {
                                int LCM = lcm(abs(Matrix[i][col]), abs(Matrix[row][col]));
                                int ta = LCM / abs(Matrix[i][col]);
                                int tb = LCM / abs(Matrix[row][col]);
                                if(Matrix[i][col] * Matrix[row][col] < 0)//异号由减变加
                                        tb = -tb;
                                for(int j = col; j < Column + 1; ++j)
                                        Matrix[i][j] = Matrix[i][j] * ta - Matrix[row][j] * tb;
                        }
                }
        }
        //row跳出时表示矩阵非零行数

        for(int i = row; i < Row; ++i)    //无解
                if(Matrix[i][Column] != 0)
                        return -1;

        if(row < Column)     //无穷多解，返回自由变元数
                return Column - row;

        for(int i = Column - 1; i >= 0; --i) {
                int temp = Matrix[i][Column];
                for(int j = i + 1;j < Column; j++)
                        if(Matrix[i][j] != 0)
                                temp -= Matrix[i][j] * X_Ans[j];
                X_Ans[i] = temp / Matrix[i][i];
        }
        return 0;
}
```

异或方程组

```cpp
#include<bits/stdc++.h>
using namespace std;
#define MAX_SIZE 350
#define ll long long
ll Matrix[MAX_SIZE][MAX_SIZE];
ll Free_x[MAX_SIZE];     //自由变元
ll X_Ans[MAX_SIZE];    //解集
ll Free_num=0;    //自由变元数

ll Guass(ll Row, ll Column)    //系数矩阵的行和列
{
        ll row = 0,col = 0, max_r;
        for(row = 0; row < Row && col < Column; row++, col++) {
                max_r = row;
                for(ll i = row + 1;i < Row; ++i)     //找出当前列最大值
                        if(abs(Matrix[i][col]) > abs(Matrix[max_r][col]))
                                max_r = i;
                if(Matrix[max_r][col] == 0) {
                        row--;
                        Free_x[Free_num++] = col + 1;
                        continue;
                }
                if(max_r!=row)    //交换
                        swap(Matrix[row], Matrix[max_r]);
                for(ll i=row+1;i<Row;i++) {
                        if(Matrix[i][col] != 0) {
                                for(ll j = col; j < Column + 1; ++j)
                                        Matrix[i][j]^=Matrix[row][j];
                        }
                }
        }
        for(ll i = row; i < Row; ++i)     //无解
                if(Matrix[i][Column] != 0)
                        return -1;

        if(row < Column)     //无穷多解
                return Column - row;

        //唯一解
        for(ll i = Column - 1; i >= 0; --i) {
                X_Ans[i] = Matrix[i][Column];
                for(ll j = i + 1; j < Column; ++j)
                        X_Ans[i] ^= (Matrix[i][j] && X_Ans[j]);
        }
        return 0;
}
```

### 线性基

```cpp
struct Linear_Basis {
        ll p[65], d[65];
        int cnt = 0;
        Linear_Basis() {
                memset(p, 0, sizeof p);
        }

        //向线性基中插入一个数
        bool ins(ll x) {
                for (int i = 62; i >= 0; --i) {
                        if (x & (1ll << i)) {
                                if (not p[i]) {
                                        p[i] = x; break;
                                }
                                x ^= p[i];
                        }
                }
                return x > 0ll;
        }

        //将线性基改造成每一位相互独立，即对于二进制的某一位i，只有pi的这一位是1，其它都是0
        void rebuild() {
                cnt = 0;
                for (int i = 62; i >= 0; --i) {
                        for (int j = i - 1; j >= 0; --j) {
                                if (p[i] & (1ll << j)) p[i] ^= p[j];
                        }
                }
                for (int i = 0; i <= 62; ++i) {
                        if (p[i]) d[++cnt] = p[i];
                }
        }

        //求线性空间与ans异或的最大值
        ll MAX(ll x) {
                for (int i = 62; i >= 0; --i) {
                        if ((x ^ p[i]) > x) x ^= p[i];
                }
                return x;
        }

        //如果是求一个数与线性基的异或最小值，则需要先rebuild，再从高位向低位依次进行异或
        ll MIN() {
                for (int i = 0; i <= 62; ++i) {
                        if (p[i]) return p[i];
                }
        }

        //求线性基能够组成的数中的第K大
        ll kth(ll k) {
                ll ret = 0;
                if (k >= (1ll << cnt)) return -1;
                for (int i = 62; i >= 0; --i) {
                        if (k & (1ll << i)) ret ^= d[i];
                }
                return ret;
        }

        //合并两个线性基
        Linear_Basis &merge (const Linear_Basis &xx) {
                for (int i = 62; i >= 0; --i) {
                        if (xx.p[i]) ins(xx.p[i]);
                }
                return *this;
        }
}LB;

//两个线性基求交 tmp不断构建A+(B\ans)
Linear_Basis merge(Linear_Basis a, Linear_Basis b) {
        Linear_Basis A = a, tmp = a, ans;
        ll cur, d;
        for (int i = 0; i <= 62; ++i) {
                if (b.p[i]) {
                        cur = 0; d = b.p[i];
                        for (int j = i; j >= 0; --j) {
                                if ((d << j) & 1) {
                                        if (tmp.p[j]) {
                                                d ^= tmp.p[j], cur ^= A.p[j];
                                                if (d) continue;
                                                ans.p[i] = cur;
                                        } else tmp.p[j] = d, A.p[j] = cur;
                                        break;
                                }
                        }
                }
        }
        return ans;
}
```

### 组合数

```cpp
ll C(int x, int y) {

        // prework
        inv[1] = inv[0] = 1; fac[1] = fac[0] = 1;
        for (int i = 2; i <= MAX; ++i) {
                inv[i] = 1ll * (mod - mod / i) * inv[mod % i] % mod;
                fac[i] = 1ll * fac[i - 1] * i % mod;
        }
        for(int i = 1; i <= MAX; ++i) {
                inv[i] = 1ll * inv[i - 1] * inv[i] % mod;
        }
        // main work
        if(y > x) return 0;
        if(x == 0 or y == 0) return 1;
        return 1ll * fac[x] * inv[y] % mod * inv[x - y] % mod;
}
```

### 间断点合并

```cpp
for (int l = 1, r, len = min(a, b); l <= len; l = r + 1) {
        r = min(a / (a / l), b / (b / l));
        ans += 1ll * (mo[r] - mo[l - 1]) * (a / l) * (b / l);
}
```

### Mobius

```cpp
// 给定整数N，求1<=x,y<=N且Gcd(x,y)为素数的数对(x,y)有多少对.
int vis[N], prime[N], num = 0, sum[N], mo[N];
void getprime() {
        mo[1] = 1;
        for (int i = 2; i <= MAX; ++i) {
                if(not vis[i]) {
                        prime[++num] = i; mo[i] = -1;
                }
                vis[i] = 1;
                for (int j = 1; j <= num and i * prime[j] <= MAX; ++j) {
                        vis[i * prime[j]] = 1;
                        if(i % prime[j] == 0) {
                                mo[i * prime[j]] = 0; break;
                        }
                        mo[i * prime[j]] = -mo[i];
                }
        }
}
int main() {
        int n = gn();
        getprime();

        // O(n) 处理 \simga p|T u(T/p) 并求前缀和
        for (int j = 1; j <= num; ++j) {
                for(int i = prime[j]; i <= MAX; i += prime[j]) {
                        sum[i] += mo[i / prime[j]];
                }
        }

        for(int i = 1; i <= n; ++i) {
                sum[i] += sum[i - 1];
        }

        ll ans = 0;
        for(int l = 1, r; l <= n; l = r + 1) {
                r = min(n, n / (n / l));
                ans += 1ll * (n / l) * (n / l) * (sum[r] - sum[l - 1]);
        }
}
```

### 整除分块

```cpp
for (int l = 1, r; l <= n; l = r + 1) {
    r = n / (n / l);
    // do something with [l, r]...
}

// 二维:
for (int l = 1, r; l <= min(n, m); l = r + 1) {
    r = min(n / (n / l), m / (m / l));
    // do something with [l, r]...
}
```

### 矩阵快速幂

```cpp

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

array<ll, 3> operator * (M a, const array<ll, 3>    b) {
        array<ll, 3> m{};
        for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                        m[j] += a.a[j][k] * b[k];
                        m[j] %= mod;
                }
        }
        return m;
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
ll Lucas(ll a, ll b) {
    if (b == 0) return 1;
    ll ret = (C(a % mod, b % mod, mod) * Lucas(a / mod, b / mod)) % mod;
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

### GCD

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
void exgcd(T a, T b, T &x, T &y){
        if(b == 0){
            x = 1, y = 0;
            return;
        }
        exgcd(b, a % b, y, x);
        y -= a / b * x;
}
```

### 求逆元

求在模$b$意义下$a^{-1}$

#### 线性求逆元

```cpp
void init() {
        inv[1] = 1;
        for (int i = 2; i <= n; ++i)
                inv[i] = (ll)(p - p / i) * inv[p % i] % p;
}
```

#### exgcd 求逆元

当$a$和$b$互质、有$exgcd(a, b, x, y)$中的 x 即为所求.

#### 费马小定理

要求$a, b$互质，并且b是质数！！！

因为 $ax \equiv 1 \pmod b$ ；

所以 $ax \equiv a^{b-1} \pmod b$；

所以 $x \equiv a^{b-2} \pmod b$ 。

### Miller Rabin 判断素数

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
            v = (ll) v * v % n;
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
const int S = 20; //随机算法判定次数，S越大，判错概率越小

//计算 (a*b)%c.     a,b都是ll的数，直接相乘可能溢出的
//    a,b,c <2^63
ll mult_mod(ll a, ll b, ll c) {
    a %= c;
    b %= c;
    ll ret = 0;
    while (b) {
        if (b & 1) {
            ret += a;
            ret %= c;
        }
        a <<= 1; //别手残，这里是a<<=1,不是快速幂的a=a*a;
        if (a >= c) a %= c;
        b >>= 1;
    }
    return ret;
}

//计算    x^n %c
ll pow_mod(ll x, ll n, ll mod) { //x^n%c
    if (n == 1) return x % mod;
    x %= mod;
    ll tmp = x;
    ll ret = 1;
    while (n) {
        if (n & 1) ret = mult_mod(ret, tmp, mod);
        tmp = mult_mod(tmp, tmp, mod);
        n >>= 1;
    }
    return ret;
}

//以a为基,n-1=x*2^t            a^(n-1)=1(mod n)    验证n是不是合数
//一定是合数返回true,不一定返回false
bool check(ll a, ll n, ll x, ll t) {
    ll ret = pow_mod(a, x, n);
    ll last = ret;
    for (int i = 1; i <= t; i++) {
        ret = mult_mod(ret, ret, n);
        if (ret == 1 && last != 1 && last != n - 1) return true; //合数
        last = ret;
    }
    if (ret != 1) return true;
    return false;
}

// Miller_Rabin()算法素数判定
//是素数返回true.(可能是伪素数，但概率极小)
//合数返回false;

bool Miller_Rabin(ll n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1) == 0) return false; //偶数
    ll x = n - 1;
    ll t = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        t++;
    }
    for (int i = 0; i < S; i++) {
        ll a = rand() % (n - 1) + 1; //rand()需要stdlib.h头文件
        if (check(a, n, x, t))
            return false; //合数
    }
    return true;
}

//************************************************
//pollard_rho 算法进行质因数分解
//************************************************
ll factor[100]; //质因数分解结果（刚返回时是无序的）
int tol; //质因数的个数。数组小标从0开始

ll gcd(ll a, ll b) {
    if (a == 0) return 1; //???????
    if (a < 0) return gcd(-a, b);
    while (b) {
        ll t = a % b;
        a = b;
        b = t;
    }
    return a;
}

ll Pollard_rho(ll x, ll c) {
    ll i = 1, k = 2, x0 = rand() % x, y = x0;
    while (1) {
        i++;
        x0 = (mult_mod(x0, x0, x) + c) % x;
        ll d = gcd(y - x0, x);
        if (d != 1 && d != x) return d;
        if (y == x0) return x;
        if (i == k) {
            y = x0;
            k += k;
        }
    }
}

//对n进行素因子分解
void findfac(ll n) {
    if (Miller_Rabin(n)) { //素数
        factor[tol++] = n;
        return;
    }
    ll p = n;
    while (p >= n) p = Pollard_rho(p, rand() % (n - 1) + 1);
    findfac(p);
    findfac(n / p);
}

// srand(time(NULL));//需要time.h头文件    //POJ上G++要去掉这句话
```

### 素数 欧拉函数

欧拉函数， $\varphi(n)$ ，表示$[1, n]$ 中和 $n$ 互质的数的个数。

- 欧拉函数是积性函数。

- $n = \sum_{d \mid n}{\varphi(d)}$ 。(莫比乌斯反演)。

- 若 $n = p^k$ ，其中 $p$ 是质数，那么 $\varphi(n) = p^k - p^{k - 1}$ 。（特别的,$\varphi(p) = p - 1$)

- 设 $n = \prod_{i=1}^{n}p_i^{k_i}$ ，其中 $p_i$ 是质数，有 $\varphi(n) = n    \prod_{i = 1}^s{\dfrac{p_i - 1}{p_i}}$ 。

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

#### 算法流程

1. 计算所有模数的积  $n$；
2. 对于第$i$个方程：
     1. 计算$m_i = \dfrac{n}{n_i}$；
     2. 计算  $m_i$  在模  $n_i$  意义下的   逆元$m_i^{-1}$ ；
     3. 计算   $c_i = m_i\times m_i^{-1}$（ **不要对  ni  取模** ）。
3. 方程组的唯一解为：$\sum_{i=1}^{k}a_ic_n(mod \ \ \ n)$  。

### 筛莫比乌斯函数

$\mu$ 为莫比乌斯函数，定义为

$$
\mu(n)=
\begin{cases}
1&n=1\\
0&n\text{ 含有平方因子}\\
(-1)^k&k\text{ 为 }n\text{ 的本质不同质因子个数}\\
\end{cases}
$$

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
}
```

### 莫比乌斯反演

设 $f(n),g(n)$ 为两个数论函数。

如果有 $f(n)=\sum_{d\mid n}g(d)$ ，那么有 $g(n)=\sum_{d\mid n}\mu(d)f(\dfrac{n}{d})$ 。

如果有 $f(n)=\sum_{n|d}g(d)$ ，那么有 $g(n)=\sum_{n|d}\mu(\dfrac{d}{n})f(d)$ 。

### 常见积性函数

PS: $F(x)$函数具有**积性**是指当$\gcd(a, b)=1$,有$F(a \times b) = F(a) \times F(b)$, **完全积性函数**没有$\gcd(a, b) = 1$的限制

- 单位函数： $\epsilon(n)=[n=1]$ （完全积性）

- 恒等函数： $\operatorname{id}_k(n)=n^k$ $\operatorname{id}_{1}(n)$ 通常简记作 $\operatorname{id}(n)$ 。（完全积性）

- 常数函数： $1(n)=1$ （完全积性）

- 除数函数： $\sigma_{k}(n)=\sum_{d\mid n}d^{k}$ $\sigma_{0}(n)$ 通常简记作 $\operatorname{d}(n)$ 或 $\tau(n)$ ， $\sigma_{1}(n)$ 通常简记作 $\sigma(n)$ 。

- 欧拉函数： $\varphi(n)=\sum_{i=1}^n [\gcd(i,n)=1]$

- 莫比乌斯函数： $\mu(n) = \begin{cases}1 & n=1 \\ 0 & \exists d>1:d^{2} \mid n \\ (-1)^{\omega(n)} & otherwise\end{cases}$ ，其中 $\omega(n)$ 表示 $n$ 的本质不同质因子个数，它也是一个积性函数。

    若 $f(x)$ 和 $g(x)$ 均为积性函数，则以下函数也为积性函数：

    $$
    \begin{aligned}
    h(x)&=f(x^p)\\
    h(x)&=f^p(x)\\
    h(x)&=f(x)g(x)\\
    h(x)&=\sum_{d\mid x}f(d)g(\dfrac{x}{d})
    \end{aligned}
    $$

## 常用公式

### 泰勒公式和二项式展开定理的共同点

对于$f(x)=(1+x)^n$，采用泰勒展开法有：
$f(x)=f_{k_0}(0)\times \frac{(x)^0}{0!}+f_{k_1}(0)\times \frac{(x)^1}{1!}+f_{k_2}(0)\times \frac{(x)^2}{2!}...$
其中$f_{k_0}(0),f_{k_1}(0).. $分别代表$f*k(x)$的$k$阶导数，并且传$0$代替$k$阶导数中的$x$,所以有：
$f*{k*0}(0)=(1+0)^n$
$f*{k*1}(0)=n\times (1+0)^{n-1}$
$f*{k_2}(0)=n\times (n-1)\times (1+0)^{n-2}$
$...$
所以有$f(x)=1^n\times \frac{x^0}{0!}+1^{n-1}\times x^1\times \frac{n}{1!}+1^{n-2}\times x^2\times \frac {n\times (n-1)}{2!}...$
联系二项式公式，可以得到上面式子与二项式展开一样。

### 欧拉函数以及 Mobius 函数的性质

欧拉函数 $\varphi(n)$ 表示小于$n$且与$n$互质的个数

1. 对于质数$n$，$\varphi(n)=n-1$
2. 对于$n = p^{k}$，$\varphi(n)=(p-1)*p^{k-1}$
3. 对于$gcd(n, m)=1$，$\varphi(n*m) = \varphi(n)\times \varphi(m)$
4. 对于$n=\prod p_{i}^{k_i}$，$\varphi(n)=n\times \prod(1-\frac{1}{p_i})$
5. 对于互质的$a$，$m$，$a^{\varphi(m)}\equiv 1(\mod m)$
6. 小于$n$且与$n$互质的数的和为$S=n\times \frac{\varphi(n)}{2}$
7. 对于质数$p$，若$n\mod p = 0$，则$\varphi(n \times p) = \varphi(n) \times p$；若$n\mod p \ne 0$，则$\varphi(n \times p) = \varphi(n) \times (p - 1)$
8. $\sum_{d|n}\varphi(d)=n$，$\varphi(n)=\sum_{d|n}\mu(d)\times \frac{n}{d}$
9. 当$n$为奇数时，$\varphi(n) = \varphi(n \times 2)$

$Mobius$函数的性质

1. $$ \sum\_{d|n}\mu(d)=\left\{ \begin{aligned} 1 &&if(n = 1) \\ 0 && if(n >1)\end{aligned} \right. $$
2. $\sum_{d|n} \frac{\mu(d)}{d}=\frac{\varphi(n)}{n}$
3. $\mu(a\times b)=\mu(a)*\mu(b)$

### 二项式定理以及指数推广到负数

$(x + 1)^n=\sum_{i=0}^{n}C(n,i)x^{i}$

$(x+1)^{-n}=\sum_{i=0}^{\infty}C(-n,i)x^{i}=\sum_{i=0}^{\infty}(-1)^{i}C(n+i-1,i)x^{i}$

$C(-n,m)=(-1)^{m}C(n+m-1,m)$

$(1-x)^{-n}=\sum_{i=0}^{\infty}C(n+i-1,i)x^{i}$

### 常用数列和公式

前 n 项平方和公式: $\dfrac{n*(n + 1)*(2n + 1)}{6}$

前 n 项立方和公式：$\dfrac{n^{2}*(n+1)^{2}}{4}$

等差数列平方和 $n*a_{1}^{2}+n*(n-1)*a_{1}*d+\dfrac{n*(n-1)*(2*n-1)*d^{2}}{6}$

### 划分问题

$n$个点最多把直线分成$C(n,0)+C(n,1)$份
$n$条直线最多把平面分成$C(n,0)+C(n,1)+C(n,2)$份
$n$个平面最多把空间分成$C(n,0)+C(n,1)+C(n,2)+C(n,3)=\dfrac{n^{3}+5*n+6}{6}$份
$n$个空间最多把时空分成$C(n,0)+C(n,1)+C(n,2)+C(n,3)+C(n,4)$份

### 约瑟夫环

N 个人围成一圈，从第一个开始报数，第 M 个将被杀掉，最后剩下一个，其余人都将被杀掉

令$f[i]$表示$i$个人玩游戏报$m$退出最后胜利者的编号 最后结果为$f[n]$

有以下递推式：

$\begin{equation} \left\{                         \begin{array}{lr}                         f[1] = 0 &                    \\                         f[i]=(f[i-1]+m)\%i & i > 1                            \end{array} \right. \end{equation}$

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

计算$(a/b)\%c$ 其中 b 能整除 a

如果$b$与$c$互素，则$(a/b)\%c=a*b^{phi(c)-1}\%c$

如果$b$与$c$不互素，则$(a/b)\%c=(a\%bc)/b$

对于$b$与$c$互素和不互素都有$(a/b)\%c=(a\%bc)/b$成立

### 因式分解

$a^{3}\pm b^{3}=(a\pm b)(a^{2}\mp ab+b^{2})$

$a^{n}-b^{n}=\left\{ \begin{array}{lr} (a-b)(a^{n-1}+a^{n-2}b+a^{n-3}b^{2}+...+ab^{n-2}+b^{n-1}) & n 为正整数\\ (a+b)(a^{n-1}+a^{n-2}b-a^{n-3}b^{2}+...+ab^{n-2}-b^{n-1})& n 为偶数 \end{array} \right. $

$a^{n}+b^{n}=(a+b)(a^{n-1}-a^{n-2}b+a^{n-3}b^{2}+...-ab^{n-2}+b^{n-1})    n为奇数$

### 三角函数

$tan(\alpha \pm \beta)=\frac{tan\alpha \pm tan\beta}{1\mp tan\alpha tan\beta}$

$\frac{a}{sinA}=\frac{b}{sinB}=\frac{c}{sinC}=2R$

$a^{2}=b^{2}+c^{2}-2bc*cosA$

$S=\frac{a*b*sinC}{2}$
