# New

## 其他

### Java String 读入

```java
class Read {
    public BufferedReader reader;
    public StringTokenizer tokenizer;

    public Read() {
        reader = new BufferedReader(new InputStreamReader(System.in));
        tokenizer = null;
    }

    public String next() {
        while (tokenizer == null || !tokenizer.hasMoreTokens()) {
            try {
                tokenizer = new StringTokenizer(reader.readLine());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return tokenizer.nextToken();
    }

    public String nextLine() {
        String str = null;
        try {
            str = reader.readLine();
        } catch (IOException e) {
            // TODO 自动生成的 catch 块
            e.printStackTrace();
        }
        return str;
    }

    public int nextInt() {
        return Integer.parseInt(next());
    }

    public long nextLong() {
        return Long.parseLong(next());
    }

    public Double nextDouble() {
        return Double.parseDouble(next());
    }

    public BigInteger nextBigInteger() {
        return new BigInteger(next());
    }
}

```
### 整数读到文件末
```cpp
public class Main {
    public static StreamTokenizer in = new StreamTokenizer(new BufferedReader(new InputStreamReader(System.in),32768));
    public static PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out));

    public static double nextDouble() throws IOException{
        in.nextToken(); return in.nval;
    }
    public static float nextFloat() throws IOException{
        in.nextToken(); return (float)in.nval;
    }
    public static int nextInt() throws IOException{
        in.nextToken(); return (int)in.nval;
    }
    public static String next() throws IOException {
        in.nextToken(); return in.sval;
    }
    public static void main(String[] args) throws IOException {
        while (in.nextToken() != StreamTokenizer.TT_EOF) {
            int h = (int) in.nval;
            System.out.printf("%d\n", qpow(3, h) - 1);
        }
    }
}
```

### 二/三分查找
```cpp
// lower_bound   find the first num >= x
//[l, mid], [mid + 1, r]
int search_one(int l, int r) {
    while (l < r) {
        int mid = (l + r) >> 1;
        if (check(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}

// upper_bound  find the first num > x
// [l, mid - 1], [mid, r]
int search_two(int l, int r) {
    while (l < r) {
        int mid = (l + r + 1) >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}

int l = 1,r = 100;
while(l < r) {
    int lmid = l + (r - l) / 3;
    int rmid = r - (r - l) / 3;
    lans = f(lmid),rans = f(rmid);
    // 求凹函数的极小值
    if(lans <= rans) r = rmid - 1;
    else l = lmid + 1;
    // 求凸函数的极大值
    if(lasn >= rans) l = lmid + 1;
    else r = rmid - 1;
}
```

## 动态规划

## 数学
### 1到n因子个数
```cpp
int val[N], vis[N], facnum[N], d[N];
vector<int> prime;
void get_facnum() {
	int pnum = 0;
	facnum[1] = 1;
	for (int i = 2; i < N; ++i) {
		if (not vis[i]) {
			prime.push_back(i);
			facnum[i] = 2;
			d[i] = 1;
		}
		for (auto to : prime) {
			if (to * i >= N) break;
			vis[to * i] = true;
			if (i % to == 0) {
				facnum[i * to] = facnum[i] / (d[i] + 1) * (d[i] + 2);
				d[i * to] = d[i] + 1;
				break;
			}
			facnum[i * to] = facnum[i] * 2;
			d[i * to] = 1;
		}
	}
}
```
### 前缀线性基
```cpp
constexpr int N = 5e5 + 100;

struct preLinear_Basis {
    array<array<int, 30>, N> p;
    array<array<int, 30>, N> pos;
    bool ins (int id, int x) {
        p[id] = p[id - 1];
        pos[id] = pos[id - 1];
        int ti = id;
        for (int i = 24; i >= 0; --i) {
            if ((x & (1 << i))) {
                if (not p[id][i]) {
                    p[id][i] = x;
                    pos[id][i] = ti;
                    break;
                }
                if (pos[id][i] < ti) {
                    swap(p[id][i], x);
                    swap(pos[id][i], ti);
                }
                x ^= p[id][i];
            }
        }

        return x > 0;
    }
    int MAX (int x, int l, int r) {
        for (int i = 24; i >= 0; --i) {
            if ((x ^ p[r][i]) > x and pos[r][i] >= l) x ^= p[r][i];
        }
        return x;
    }
}LB;

int main() {
    int n = gn();
    for (int i = 1; i <= n; ++i) {
        int val = gn();
        LB.ins(i, val);
    }
    int q = gn();
    while (q--) {
        int l = gn(), r = gn();
        cout << LB.MAX(0, l, r) << endl;
    }
}
```

### FFT

```cpp
const int N = 3e6 + 7;
const double PI = acos(-1);

int limit = 1, L, R[N];  // n < limit, 二进制位数, 蝴蝶变换

struct Complex {
  double x, y;
  Complex(double x = 0, double y = 0) : x(x), y(y) {}
} a[N], b[N];

Complex operator*(Complex A, Complex B) {
  return Complex(A.x * B.x - A.y * B.y, A.x * B.y + A.y * B.x);
}
Complex operator-(Complex A, Complex B) {
  return Complex(A.x - B.x, A.y - B.y);
}
Complex operator+(Complex A, Complex B) {
  return Complex(A.x + B.x, A.y + B.y);
}

void FFT(Complex *A, int type) { // type:   1: DFT, -1: IDFT
  for (int i = 0; i < limit; ++i)
    if (i < R[i]) swap(A[i], A[R[i]]);   // 防止重复

  for (int mid = 1; mid < limit; mid <<= 1) {
    //待合并区间长度的一半，最开始是两个长度为1的序列合并,mid = 1;
    Complex wn(cos(PI / mid), type * sin(PI / mid));  //单位根w_n^1;

    for (int len = mid << 1, pos = 0; pos < limit; pos += len) {
      // len是区间的长度，pos是当前的位置,也就是合并到了哪一位
      Complex w(1, 0);  //幂,一直乘，得到平方，三次方...

      for (int k = 0; k < mid; ++k, w = w * wn) {
        //只扫左半部分，蝴蝶变换得到右半部分的答案,w 为 w_n^k
        Complex x = A[pos + k];            //左半部分
        Complex y = w * A[pos + mid + k];  //右半部分
        A[pos + k] = x + y;                //左边加
        A[pos + mid + k] = x - y;          //右边减
      }
    }
  }
  if (type == 1) return;
  for (int i = 0; i <= limit; ++i) a[i].x /= limit;
  //最后要除以limit也就是补成了2的整数幂的那个N，将点值转换为系数
  //（前面推过了点值与系数之间相除是N）
}

int main() {
  int n = gn<int>(), m = gn<int>();
  for (int i = 0; i <= n; ++i) a[i].x = gn<int>();
  for (int i = 0; i <= m; ++i) b[i].x = gn<int>();
  while (limit <= n + m) limit <<= 1, L++;
  //也可以写成：limit = 1 << int(log2(n + m) + 1);
  // 补成2的整次幂，也就是N
  for (int i = 0; i < limit; ++i)
    R[i] = (R[i >> 1] >> 1) | ((i & 1) << (L - 1));
  FFT(a, 1);  // FFT 把a的系数表示转化为点值表示
  FFT(b, 1);  // FFT 把b的系数表示转化为点值表示
  //计算两个系数表示法的多项式相乘后的点值表示
  for (int i = 0; i <= limit; ++i) a[i] = a[i] * b[i];
  //对应项相乘，O(n)得到点值表示的多项式的解C，利用逆变换完成插值得到答案C的点值表示
  FFT(a, -1);

  for (int i = 0; i <= n + m; ++i)
    printf("%d ", (int)(a[i].x + 0.5));  //注意要+0.5，否则精度会有问题
}

```

### NTT

```cpp
const int MOD = 998244353, G = 3, Gi = 332748118;  //这里的Gi是G的除法逆元
const int N = 5000007;
const double PI = acos(-1);

int n, m, res, limit = 1;  //
int L;          //二进制的位数
int RR[N];
ll a[N], b[N];

void NTT(ll *A, int type) {
  for (int i = 0; i < limit; ++i)
    if (i < RR[i]) swap(A[i], A[RR[i]]);
  for (int mid = 1; mid < limit; mid <<= 1) {  //原根代替单位根
    // ll wn = qpow(type == 1 ? G : Gi, (MOD - 1) / (mid << 1));
    ll wn = qpow(G, (MOD - 1) / (mid * 2));
    if (type == -1) wn = qpow(wn, MOD - 2);
    //逆变换则乘上逆元,因为我们算出来的公式中逆变换是(a^-ij)，也就是(a^ij)的逆元
    for (int len = mid << 1, pos = 0; pos < limit; pos += len) {
      ll w = 1;
      for (int k = 0; k < mid; ++k, w = (w * wn) % MOD) {
        int x = A[pos + k], y = w * A[pos + mid + k] % MOD;
        A[pos + k] = (x + y) % MOD;
        A[pos + k + mid] = (x - y + MOD) % MOD;
      }
    }
  }

  if (type == -1) {
    ll limit_inv = inv(limit);  // N的逆元（N是limit, 指的是2的整数幂）
    for (int i = 0; i < limit; ++i)
      a[i] =
          (a[i] * limit_inv) %
          MOD;  // NTT还是要除以n的，但是这里把除换成逆元了，inv就是n在模MOD意义下的逆元
  }
}  //代码实现上和FFT相差无几
//多项式乘法
void poly_mul(ll *a, ll *b, int deg) {
  for (limit = 1, L = 0; limit <= deg; limit <<= 1) L++;
  for (int i = 0; i < limit; ++i) {
    RR[i] = (RR[i >> 1] >> 1) | ((i & 1) << (L - 1));
  }
  NTT(a, 1);
  NTT(b, 1);
  for (int i = 0; i < limit; ++i) a[i] = a[i] * b[i] % MOD;
  NTT(a, -1);
}

int main() {
  n = gn(), m = gn();
  for (int i = 0; i <= n; ++i) a[i] = (gn() + MOD) % MOD;  //取模好习惯
  for (int i = 0; i <= m; ++i) b[i] = (gn() + MOD) % MOD;
  poly_mul(a, b, n + m);
  for (int i = 0; i <= n + m; ++i) printf("%d ", a[i]);
  return 0;
}

```

预处理版本

```cpp
const int N = 5e6+7;
const int MOD = 998244353;

int qpow(int a, int b) {
  int res = 1;
  while (b) {
    if (b & 1) res = 1ll * res * a % MOD;
    a = 1ll * a * a % MOD;
    b >>= 1;
  }
  return res;
}

namespace Poly {
  typedef vector<int> poly;
  const int G = 3;
  const int inv_G = qpow(G, MOD - 2);
  int RR[N], deer[2][22][N], inv[N];

  void init(const int t) {  //预处理出来NTT里需要的w和wn，砍掉了一个log的时间
    for (int p = 1; p <= t; ++p) {
      int buf1 = qpow(G, (MOD - 1) / (1 << p));
      int buf0 = qpow(inv_G, (MOD - 1) / (1 << p));
      deer[0][p][0] = deer[1][p][0] = 1;
      for (int i = 1; i < (1 << p); ++i) {
        deer[0][p][i] = 1ll * deer[0][p][i - 1] * buf0 % MOD;  //逆
        deer[1][p][i] = 1ll * deer[1][p][i - 1] * buf1 % MOD;
      }
    }
    inv[1] = 1;
    for (int i = 2; i <= (1 << t); ++i)
      inv[i] = 1ll * inv[MOD % i] * (MOD - MOD / i) % MOD;
  }

  int NTT_init(int n) {
    int limit = 1, L = 0;
    while (limit < n) limit <<= 1, L++;
    for (int i = 0; i < limit; ++i)
      RR[i] = (RR[i >> 1] >> 1) | ((i & 1) << (L - 1));
    return limit;
  }

  #define ck(x) (x >= MOD ? x - MOD : x)

  void NTT(poly &A, int type, int limit) { // 1: DFT, 0: IDFT
    A.resize(limit);
    for (int i = 0; i < limit; ++i)
      if (i < RR[i]) swap(A[i], A[RR[i]]);
    for (int mid = 2, j = 1; mid <= limit; mid <<= 1, ++j) {
      int len = mid >> 1;
      for (int pos = 0; pos < limit; pos += mid) {
        int *wn = deer[type][j];
        for (int i = pos; i < pos + len; ++i, ++wn) {
          int tmp = 1ll * (*wn) * A[i + len] % MOD;
          A[i + len] = ck(A[i] - tmp + MOD);
          A[i] = ck(A[i] + tmp);
        }
      }
    }
    if (type == 0) {
      int inv_limit = qpow(limit, MOD - 2);
      for (int i = 0; i < limit; ++i) A[i] = 1ll * A[i] * inv_limit % MOD;
    }
  }

  poly poly_mul(poly A, poly B) {
    int deg = A.size() + B.size() - 1;
    int limit = NTT_init(deg);
    poly C(limit);
    NTT(A, 1, limit);
    NTT(B, 1, limit);
    for (int i = 0; i < limit; ++i) C[i] = 1ll * A[i] * B[i] % MOD;
    NTT(C, 0, limit);
    C.resize(deg);
    return C;
  }
}  // namespace Poly

using Poly::poly;
using Poly::poly_mul;

int n, m, x;
poly f, g;

int main() {
  Poly::init(21);
  n = gn();
  m = gn();
  for (int i = 0; i < n + 1; ++i) x = gn(), f.push_back(x + MOD % MOD);
  for (int i = 0; i < m + 1; ++i) x = gn(), g.push_back(x + MOD % MOD);

  g = poly_mul(f, g);
  for (int i = 0; i < n + m + 1; ++i) printf("%d ", g[i]);
  return 0;
}

```

### exgcd 求逆元

```cpp
template <class T>
T exgcd(T a, T b, T &x, T &y) {
    if (!b) {
        x = 1, y = 0;
        return a;
    }
    T t, ret;
    ret = exgcd(b, a % b, x, y);
    t = x, x = y, y = t - a / b * y;
    return ret;
}

template<typename T>
T inv(T num, T mod) {
    T x, y;
    exgcd(num, mod, x, y);
    return x;
}
```

## 计算几何

### 基本

```cpp
// 极坐标 极角 (-pi, pi]
double theta(double x, double y) {
    if (x > 0) return atan(y/x);

    if (x == 0) {
        if (y > 0) return pi/2;
        return -pi/2;
    } else {
        if (y >= 0) return atan(y/x) + pi;
        return atan(y/x) - pi;
    }

}
```

## 图论

### 2-SAT

$i,j$不能同时选：选了$i$就要选$j′$，选$j$就要选$i′$。故$i→j′,j→i′$。一般操作即为$a_i\space xor\space a_j=1$
$i,j$必须同时选：选了$i$就要选$j$，选$j$就要选$i$。故$i→j,j→i$。一般操作即为$a_i\space xor\space a_j=0$
$i,j$任选（但至少选一个）选一个：选了$i$就要选$j′$，选$j$就要选$i′$，选$i′$就要选$j$，选$j′$就要选$i$。故$i→j′,j→i′,i′→j,j′→i$。一般操作即为$a_i\space or\space a_j=1$
$i$必须选：直接$i′→i$，可以保证无论怎样都选$i$。一般操作为给出的$a_i=1$或$a_i\space and\space a_j=1$

```cpp
constexpr int N = 2e5 + 100;

int n, m;
bool vis[N];

struct node {
    int to, nxt;
} s[N];

int head[N], tot = 0, id[N];

void add(int x, int y) {
    s[++tot] = {y, head[x]};
    head[x] = tot;
}

stack<int> st;

bool dfs(int node) {
    if (vis[node ^ 1]) return false;
    if (vis[node]) return true;
    vis[node] = true;
    st.push(node);
    for (int i = head[node]; ~i; i = s[i].nxt) {
        if (not dfs(s[i].to)) return false;
    }
    return true;
}

bool solve() {
    for (int i = 2; i <= 2 * n; i += 2) {
        if (not vis[i] and not vis[i ^ 1]) {
            if (not dfs(i)) {
                while (not st.empty()) {
                    vis[st.top()] = false;
                    st.pop();
                }

                if (not dfs(i + 1)) return false;
            }
        }
    }

    return true;
}

// 2 3  4 5
// R pre B last

//有k个灯，n个嘉宾，每个嘉宾会选择3个灯进行猜颜色（只有红色和蓝色），猜中两个以上有奖，问怎么设置灯的颜色能使所有嘉宾都能得奖。
int main() {
    memset(head, -1, sizeof head);
    n = gn(), m = gn();
    for (int i = 1; i <= n; ++i) id[i] = 2 * i;
    for (int i = 1; i <= m; ++i) {
        vector<pair<int, char> > v;
        for (int j = 0; j < 3; ++j) {
            int val = gn(); char c;
            scanf(" %c", &c);
            v.emplace_back(val, c);
        }

        int nowid;
        nowid = id[v[0].first] + (v[0].second == 'R');
        add(nowid, id[v[1].first] + (v[1].second == 'B'));
        add(nowid, id[v[2].first] + (v[2].second == 'B'));

        nowid = id[v[1].first] + (v[1].second == 'R');
        add(nowid, id[v[0].first] + (v[0].second == 'B'));
        add(nowid, id[v[2].first] + (v[2].second == 'B'));

        nowid = id[v[2].first] + (v[2].second == 'R');
        add(nowid, id[v[1].first] + (v[1].second == 'B'));
        add(nowid, id[v[0].first] + (v[0].second == 'B'));

    }

    if (solve()) {
        for (int i = 2; i <= 2 * n; i += 2) {
            if (vis[i]) cout << 'R';
            else cout << 'B';
        }
        cout << '\n';
    } else puts("-1");
}
```

## 数据结构

### cdq 分治

```cpp
void cdqmax(int l, int r) {
    if (l == r) return ;
    int mid = (l + r) >> 1;
    // 递归处理左右
    cdqmax(l, mid);
    cdqmax(mid + 1, r);

    // 合并
    merge(s + l, s + mid + 1, s + mid + 1, s + r + 1, tem + l, cmpa);
    for (int i = l; i <= r; ++i) {
        s[i] = tem[i];
    }

    // 计算贡献
    for (int i = l; i <= r; ++i) {
        if (s[i].type == 0 && s[i].tim <= mid) {
            tree.change(s[i].l, s[i].h);
        }
        if (s[i].type == 1 && s[i].tim > mid) {
            int maxn = tree.querymax(s[i].l, s[i].r);
            if (maxn == 0) continue;
            if (ANS[s[i].id] == -1) ANS[s[i].id] = s[i].h - maxn;
            else ANS[s[i].id] = min(ANS[s[i].id], s[i].h - maxn);
        }
    }

    // 消除影响
    for (int i = l; i <= r; ++i) {
        if (s[i].type == 0 && s[i].tim <= mid) {
            tree.change(s[i].l, 0);
        }
    }
}
```
### KD-tree
```cpp
constexpr int N = 2e5 + 100;

#define cmax(a, b) (a < b ? a = b : a)
#define cmin(a, b) (a > b ? a = b : a)
#define ls t[mid].s[0]
#define rs t[mid].s[1]
constexpr ll INF = 1e18;

ll sqr(int x) {return 1ll * x * x;}

int D, root;

struct P {
    int d[3], id;
    bool operator < (const P & rhs) const {
        return d[D] < rhs.d[D];
    }
}a[N];

struct kd_node {
    int d[3], s[2], x[2], y[2], z[2], id;
} t[N];

void update(int f, int x) {
    cmin(t[f].x[0], t[x].x[0]), cmax(t[f].x[1], t[x].x[1]);
    cmin(t[f].y[0], t[x].y[0]), cmax(t[f].y[1], t[x].y[1]);
    cmin(t[f].z[0], t[x].z[0]), cmax(t[f].z[1], t[x].z[1]);
}

int build(int l, int r, int d) {
    D = d; int mid = (l + r) >> 1;
    nth_element(a + l, a + mid, a + r + 1);
    t[mid].d[0] = t[mid].x[0] = t[mid].x[1] = a[mid].d[0];
    t[mid].d[1] = t[mid].y[0] = t[mid].y[1] = a[mid].d[1];
    t[mid].d[2] = t[mid].z[0] = t[mid].z[1] = a[mid].d[2];
    t[mid].id = a[mid].id;
    if (l < mid) ls = build(l, mid - 1, d ^ 1), update(mid, ls);
    if (r > mid) rs = build(mid + 1, r, d ^ 1), update(mid, rs);
    return mid;
}

ll getdist(int node, int x, int y, int z) {
    // 曼哈顿距离 min
//        return max(t[node].x[0] - x, 0) + max(x - t[node].x[1], 0) + max(t[node].y[0] - x, 0) + max(x - t[node].y[1], 0);
    // max max(abs(x - t[node].x[1]), abs(t[node].x[0] - x)) + max(abs(y - t[node].y[1]), abs(t[node].y[0] - y))
    // 欧几里得距离
    // min sqr(max({x - t[node].x[1], t[node].x[0] - x, 0})) + sqr(max({y - t[node].y[1], t[node].y[0] - y, 0}))
     if (t[node].z[0] > z) return 1e18;
     return sqr(max({x - t[node].x[1], t[node].x[0] - x, 0})) + sqr(max({y - t[node].y[1], t[node].y[0] - y, 0}));
    // max max(sqr(x - t[node].x[0]), sqr(t[node].x[0] - x)) + max(sqr(y - t[node].y[1]), sqr(t[node].y[0] - y))
}

void insert(int node, int x, int y, int z, int id) {
    t[node].d[0] = t[node].x[0] = t[node].x[1] = x;
    t[node].d[1] = t[node].y[0] = t[node].y[1] = y;
    t[node].d[2] = t[node].z[0] = t[node].z[1] = z;
    t[node].id = id;
    for (int p = root, k = 0; p; k ^= 1) {
        update(p, node);
        int &nxt = t[p].s[t[node].d[k] > t[p].d[k]];
        if (nxt == 0) {
            nxt = node;
            return;
        } else p = nxt;
    }
}

void query(int node, ll &ans, int &id, int x, int y, int z) {
    ll tmp = t[node].d[2] <= z ? sqr(t[node].d[0] - x) + sqr(t[node].d[1] - y) : 1e18, d[2];
//    cout << node << ' ' << t[node].d[0] << ' ' << t[node].d[1] << ' ' << t[node].d[2] << ' ' << ans << ' ' << tmp << endl;
    if (t[node].s[0]) d[0] = getdist(t[node].s[0], x, y, z); else d[0] = INF;
    if (t[node].s[1]) d[1] = getdist(t[node].s[1], x, y, z); else d[1] = INF;

    if (tmp < ans) {
        ans = tmp, id = t[node].id;
    } else if (tmp == ans and id > t[node].id) id = t[node].id;

    tmp = d[0] >= d[1];
    if (d[tmp] <= ans) query(t[node].s[tmp], ans, id, x, y, z);
    tmp ^= 1;
    if (d[tmp] <= ans) query(t[node].s[tmp], ans, id, x, y, z);
}

struct node {
    int x, y, c, id;
}hotel[N], guest[N];

void solve() {
   int n = gn(), m = gn();
   for (int i = 1; i <= n; ++i) {
       int x = gn(), y = gn(), c = gn();
       hotel[i] = {x, y, c, i};
   }

    for (int i = 1; i <= m; ++i) {
        int x = gn(), y = gn(), c = gn();
        guest[i] = {x, y, c, i};
    }

    for (int i = 1; i <= n; ++i) a[i] = {hotel[i].x, hotel[i].y, hotel[i].c, hotel[i].id};

    ll val; int idx;
    root = build(1, n, 0);

    for (int i = 1; i <= m; ++i) {
        val = INF, idx = 0x3f3f3f3f;
        query(root, val, idx, guest[i].x, guest[i].y, guest[i].c);
        cout << hotel[idx].x << ' ' << hotel[idx].y << ' ' << hotel[idx].c << '\n';
    }
}
```

## 字符串

### 最小表示法

求循环同构的最小表示

```cpp
int a[N];

int min_show(int n) {
    int i = 0, j = 1, k = 0;
    while (i < n and j < n and k < n) {
        if (a[(i + k) % n] == a[(j + k) % n]) ++k;
        else {
            if (a[(i + k) % n] > a[(j + k) % n]) i += k + 1;
            else j += k + 1;
            if (i == j) ++i;
            k = 0;
        }
    }
    return min(i, j);
}

int main() {
    int n = gn();
    for (int i = 0; i < n; ++i) {
        a[i] = gn();
    }
    int idx = min_show(n);
//    cout << idx << endl;
    int cnt = 0;
    while (cnt < n) {
        ++cnt;
        cout << a[idx] << " \n"[cnt == n];
        idx = (idx + 1) % n;
    }
}
```
