[TOC]

## 其他

### 阴间快读

```cpp
struct fastin {
    char buf[BUFSIZ], *it, *en;
    char getchar() {
        if (it == en) it = buf, en = buf + fread(buf, 1, BUFSIZ, stdin);
        return *it++;
    }
    template<typename T>
    fastin& operator >> (T &i) {
        int f; char c = getchar();
        for (f = 1;!isdigit(c); c = getchar()) if (c == '-') f = -1;
        for (i = 0; isdigit(c); c = getchar()) i = i * 10 + c - '0';
        i *= f; return *this;
    }
} fin;
```

### 龟速乘

```cpp
ll mulmod( ll a , ll b ) {
    ll ans = 0 ;
    for( ; b ; b >>= 1) {
        if ( b & 1) ans = ( ans + a ) % mod ;
        a = a ∗ 2 % mod ;
    }
    return ans ;
}
```

### 快速幂

```jsx
ll qpow(ll b) {
    ll ans = 1, sign = 2;
    while(b) {
        if(b & 1) ans = (ans * sign) % mod;
        sign = (sign * sign) % mod;
        b >>= 1;
    }
    return ans % mod;
}
```

### 三分

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
double x[50005], y[50005];
const double eps = 1e-7;
double check(double now){
    double cnt = 0.0;
    for(int i = 1; i <= n; i++) cnt = max(cnt, y[i] * y[i] + (x[i] - now) * (x[i] - now));
    return cnt;
}

int main(){
    while(scanf("%d", &n)){
        if(n == 0) break;
        double L = 2000000.0, R = -2000000.0;
        for(int i = 1; i <= n; i++){
            scanf("%lf%lf", &x[i], &y[i]);
            L = min(x[i], L);
            R = max(x[i], R);
        }
        double mid, fuck, sf;
        while(1){
            sf = (R - L) / 3;
            mid = L + sf;
            fuck = L + 2 * sf;
            if(check(mid) > check(fuck)) L = mid;
            else R = fuck;
            if(R - L < eps) break;
        }
        if( fabs(L) < eps ) L = 0;
        printf("%.9lf %.9lf\n", L, sqrt(check(L)));
    }
    return 0;
}
```

### 快读快输

```cpp
template<typename T>
inline T read(){
    T s = 0,f = 1; char ch = getchar();
    while(!isdigit(ch)) {if(ch == '-') f = -1;ch = getchar();}
    while(isdigit(ch)) {s = (s << 3) + (s << 1) + ch - 48; ch = getchar();}
    return s * f;
}
#define gn() read<int>()
#define gl() read<ll>()

template<typename T>
inline void print(T x) {
    if(x < 0) putchar('-'), x = -x;
    if(x > 9) print(x / 10);
    putchar(x % 10 + '0');
}
```

### Java高精度计算和输入输出

```java
import java.math.BigInteger;
import java.math.BigDecimal;
import java.util.*;
public class Main {

	public static void main(String[] args) {
		//创建方式
		//方式1 Scanner读入
		Scanner in = new Scanner(System.in); 
		while(in.hasNext()) //等同于!=EOF
		{
		    BigInteger a;
		    a = in.nextBigInteger();
		    System.out.print(a.toString());
		}
		//方式2 初始化创建
		String str = "123";
		BigInteger a = BigInteger.valueOf(str);

		int num = 456;
		BigInteger a = BigInteger.valueOf(num);
		
		//p进制输出
		int p;
		System.out.print(a.toString(p)); // 输出a的p进制，不填默认10进制

		//比较
		BigInteger a = new BigInteger("123");
		BigInteger b = new BigInteger("456");

		System.out.println(a.equals(b)); // a == b 时为 true 否则为 false
		if(a.compareTo(b) == 0) System.out.println("a == b"); // a == b
		else if(a.compareTo(b) > 0) System.out.println("a > b"); // a > b
		else if(a.compareTo(b) < 0) System.out.println("a < b"); // a < b

		BigInteger c = a.add(b); //加法 记得赋值
		BigInteger c = a.subtract(b); //减法
		BigInteger c = a.multiply(b); //乘法
		BigInteger c = a.divide(b);//除法
		BigInteger c = b.remainder(a);//取余
		BigInteger c = a.abs();//取绝对值
		BigInteger c = a.negate();//取负数
		BigInteger POW = a.pow();//幂
		BigInteger GCD = a.gcd()//最大公约数
		
		//创建对象
		BigDecimal b1 = new BigDecimal("1.34");//1.34
		BigDecimal b2 = BigDecimal.valueOf(1.34);//1.34

		//读入

		BigDecimal b3 = in.nextBigDecimal();
		//加,减，乘，除，转换

		BigDecimal c4 = b1.add(b2);
		BigDecimal c4 = b1.subtract(b2);
		BigDecimal c4 = b1.multiply(b2);
		BigDecimal c4 = b1.divide(b2);
		double d = b1.doubleValue();
		String s = b1.toString();
		long num = b1.longValue();
		int num = b1.longValue();

		//比较
		int a = bigdemical.compareTo(bigdemical2);
		//-1表示bigdemical < bigdemical2
		//0表示bigdemical == bigdemical2
		//1表示bigdemical > bigdemical2

		//设置精度
		//scale表示精度几位 
		//ROUND_HALF_UP 四舍五入
		//ROUND_DOWN 直接截断
		//ROUND_UP 向上取
		b.setScale(scale, BigDecimal.ROUND_HALF_UP);
		
		//快读
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
		public static String next() throws IOException{ 
            in.nextToken(); return in.sval;
        }

		public static void main(String[] args) throws IOException{
		//		获取输入
            while(in.nextToken()!=StreamTokenizer.TT_EOF){
                break;
            }
            int x = (int)in.nextToken();  //第一个数据应当通过nextToken()获取

            int y = nextInt();
            float f = nextFloat();
            double d = nextDouble();
            String str = next();

            //快速输出
            out.println("abc");
            out.flush();
            out.close();
        }
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



## 博弈

### Bash 博弈 

有一堆石子共有$N$个。$A,B$两个人轮流拿，$A$ 先拿。每次最少拿$1$颗，最多拿$K$颗，拿到最后$1$颗石子的人获胜。假设$A,B$都非常聪明，拿石子的过程中不会出现失误。给出 $N$ 和 $K$，问最后谁能赢 得比赛。

先手必胜 当且仅当 $N\%(K + 1) = 0$ 

### Nim 博弈 

有 $N$ 堆石子。$A,B$ 两个人轮流拿，$A$ 先拿。每次只能从一堆中取若干个，可将一堆全取走，但不可不取，拿到最后$1$颗石子的人获胜。假设$A,B$都非常聪明，拿石子的过程中不会出现失误。给出$N$及每堆石子的数量，问最后谁能赢得比赛。 

先手必胜 当且仅当 $X1\bigoplus X2\bigoplus ……\bigoplus Xn \neq 0$ 

### Wythoff 博弈 

有$2$堆石子。$A,B$ 两个人轮流拿，$A$先拿。每次可以从一堆中取任意个或从 $2$ 堆中取相同数量的 石子，但不可不取。拿到最后 $1$ 颗石子的人获胜。假设 $A,B$ 都非常聪明，拿石子的过程中不会出现失误。给出 $2$ 堆石子的数量，问最后谁能赢得比赛。

```cpp
void Wythoff(int n, int m) {
    if(n > m) swap(n, m);
    int tmp = (m - n) * (sqrt(5) + 1.0) / 2;
    if(n == tmp) puts("B");
    else puts("A");
}
```

### 公平组合游戏 

若一个游戏满足： 

1. 游戏由两个人参与，二者轮流做出决策
2. 在游戏进程的任意时刻，可以执行的合法行动与轮到哪名玩家无关 
3.  有一个人不能行动时游戏结束 则称这个游戏是一个公平组合游戏NIM 游戏就是一个 公平组合游戏 
### SG-组合游戏 
一个公平组合游戏若满足： 
1. 两人的决策都对自己最有利
2. 当有一人无法做出决策时游戏结束，无法做出决策的人输，且游戏一定能在有限步数内结束 
3. 游戏中的同一个状态不可能多次抵达，且游戏不会出现平局 则这类游戏可以用 SG 函数解决，我们称之为 SG-组合游戏 
### 删边游戏 
1. 树的删边游戏 给出一个有 $N$ 个点的树，有一个点作为树的根节点。 游戏者轮流从树中删边，删去一条边后，不与根节点相连的部分将被移走。 无法行动者输。 有如下定理：叶子节点的 $SG$ 值为 0；其它节点的 $SG$ 值为它的所有子节点的 $SG$ 值加 1 后的异或 和。 

2. 无向图删边游戏 一个无向连通图，有一个点作为图的根。 游戏者轮流从图中删去边，删去一条边后，不与根节点相连的部分被移走，无法行动者输。 
$Fusion Principle$ ： 我们可以对无向图做如下改动：将图中的任意一个偶环缩成一个新点，任意一个奇环缩成一个新 点加一个新边；所有连到原先环上的边全部改为与新点相连。这样的改动不会影响图的 SG 值。 这样我们就可以将任意一个无向图改成树结构。

## 字符串

### 常见题型总结

#### 单字符串问题

1.可重叠最长重复子串 

解法：
${height}$数组最大值。若需要输出则输出后缀子串$sa[i-1]$和$sa[i]$最长公共前缀

2.不可重叠最长重复子串

二分答案，每次check按照$height ≥ K$ 分组，判断组内的$sa$最大值与最小值之差是否大于等于K

3.可重叠的至少出现K次的最长重复子串

解法： 二分最长子串长度，每次check 按照$height >=K$的分组，判断是否出现K次。

4.至少出现两次的子串个数

$ans = \sum _{i = 2} ^ {n} max(height[i] - height[i - 1], 0)$

5.字符串的不相同子串个数

每个后缀k会产生$n - sa[k]$个前缀，但是由于重复计数所以需要$n - sa[k] - height[i]$

6.字符串字典序排第K的子串

7.最长回文串

把原串倒过来接到原串上 两者之间插入特殊字符'#' 变成两个串的最长公共前缀问题

### 字符串Hash

#### 单Hash

```cpp
/*
模数备用
1610612741 > 402653189 > 201326611 > 805306457 > 50331653 > 6291469 > 
19260817 > 12582917 > 786433 > 196613
*/
struct My_hash {
    static const int maxn = 1e5 + 100;
    typedef unsigned long long ull;
    static const int base = 131;
    ull p[maxn], hs[maxn];
    void getp() {
        p[0] = 1;
        for(int i = 1; i < maxn; ++i) {
            p[i] = p[i - 1] * base;
        }
    }
    void SetHash(string s) {
        int n = s.length();
        s = "*" + s;
        for(int i = 1; i <= n; ++i) {
            hs[i] = hs[i - 1] * base + (ull)s[i];
        }
    }
    ull GetHash(int l, int r) {
        return (ull)hs[r] - p[r - l + 1] * hs[l - 1];
    }
};
```

#### 双Hash法

```cpp
struct My_hash {
    static const int maxn = 1e5 + 100;
    typedef unsigned long long ull;
    static const ull baseone = 131;
    static const ull basetwo = 233;
    static const ull modone = 1e9 + 7;
    static const ull modtwo = 1e9 + 9;
    ull pone[maxn], hsone[maxn], ptwo[maxn], hstwo[maxn];
    void getp() {
        pone[0] = ptwo[0] = 1;
        for(int i = 1; i < maxn; ++i) {
            pone[i] = pone[i - 1] * baseone % modone;
            ptwo[i] = ptwo[i - 1] * basetwo % modtwo;
        }
    }
    void SetHash(string s) {
        int n = s.length();
        s = "*" + s;
        for(int i = 1; i <= n; ++i) {
            hsone[i] = (hsone[i - 1] * baseone % modone + (ull)s[i]) % modone;
            hstwo[i] = (hstwo[i - 1] * basetwo % modtwo + (ull)s[i]) % modtwo;
        }
    }
    pair<ull, ull> GetHash(int l, int r) {
        ull tmpone = ((ull)hsone[r] - pone[r - l + 1] * hsone[l - 1] % modone + modone) % modone;
        ull tmptwo = ((ull)hstwo[r] - ptwo[r - l + 1] * hstwo[l - 1] % modtwo + modtwo) % modtwo;
        return {tmpone, tmptwo} ;
    }
};

```

### Trie树

```cpp
struct DictionaryTree {
    static const int maxn = 2e5 + 100;
    int tree[maxn][26], tot = 0;
    bool isend[maxn];
    void insert(char *s) {
        int len = strlen(s);
        int root = 0;
        for(int i = 0; i < len; ++i) {
            int id = s[i] - 'a';
            if(!tree[root][id]) {
                tree[root][id] = ++tot;
            }
            root = tree[root][id];
        }
        isend[root] = true;
    }

    bool find(char *s) {
        int len = strlen(s);
        int root = 0;
        for(int i = 0; i < len; ++i) {
            int id = s[i] - 'a';
            if(!tree[root][id]) return false;
            root = tree[root][id];
        }
        return isend[root];
    }
};
```

### KMP

```cpp
namespace KMP{
    vector<int> next;

    void build(const string &pattern){
        int n = pattern.length();
        next.resize(n + 1);
        for (int i = 0, j = next[0] = -1; i < n; next[++i] = ++j){
            while(~j && pattern[j] != pattern[i]) j = next[j];
        }
    }

    vector<int> match(const string &pattern, const string &text){
    	//res返回所有匹配位置 
        vector<int> res;
        int n = pattern.length(), m = text.length();
        build(pattern);
        for (int i = 0, j = 0; i < m; ++i){
            while(j > 0 && text[i] != pattern[j]) j = next[j];
            if (text[i] == pattern[j]) ++j;
            if (j == n) res.push_back(i - n + 1), j = next[j];
        }
        return res;
    }
};

struct KMP {
    static const int N = 1e5 + 100;
    int net[N];
    void GetNext(string p) {
        int plen = p.length();
        int i = 0, j = -1;
        net[0] = -1;
        while(i < plen - 1) {
            if(j == -1 || p[i] == p[j]) {
                ++i, ++j;
                net[i] = j;
            } else j = net[j];
        }
    }
    int match(string s, string p) {
        GetNext(p);
        int slen = s.length();
        int plen = p.length();
        int i = 0, j = 0;
        while(i < slen && j < plen) {
            if(j == -1 || s[i] == p[j]) {
                ++i, ++j;
            } else j = net[j];
        }
        if(j == plen) return i - j;
        return -1;
    }
};

```

### manacher

```cpp
struct Manacher {
    vector<int> ans, str;
    int build(const string &s) { 
        int n = s.length(), m = (n + 1) << 1, ret = 0;
        str.resize(m + 1), ans.resize(m + 1);
        str[0] = '$', str[m] = '@', str[1] = '#';
        ans[1] = 1;
        for(int i = 1; i <= n; ++i) {
            str[i << 1] = s[i - 1]; str[i << 1 | 1] = '#';
        }
        for(int r = 0, p = 0, i = 2; i < m; ++i) {
            if(r > i) ans[i] = min(r - i, ans[p * 2 - i]);
            else ans[i] = 1;
            while(str[i - ans[i]] == str[i + ans[i]]) ++ans[i];
            if(i + ans[i] > r) r = i + ans[i], p = i;
            ret = max(ret, ans[i] - 1);
        }
        return ret;
    }
    int mid (int x, bool odd) {
        if(odd) return ans[(x + 1) << 1] - 1;
        return ans[(x + 1) << 1 | 1] - 1;
    }
};

```

### AC自动机

```cpp
//匹配出现过的模式串：
struct ACAM {
    static const int maxn = 1e6 + 100;
    int tot = 0;
    struct node {
        int net[26];
        int fail, cnt;
    }trie[maxn];
    void init() {
        for(int i = 0; i < maxn; ++i) {
            memset(trie[i].net, 0, sizeof(trie[i].net));
            trie[i].cnt = trie[i].fail = 0;
        }
        tot = 0;
    }
	// build Trie Tree
    void insert(const string &s, int cnt) {
        int root = 0;
        for(int i = 0, len = s.length(); i < len; ++i) {
            if(!trie[root].net[s[i] - 'a']) {
                trie[root].net[s[i] - 'a'] = ++tot;
            }
            root = trie[root].net[s[i] - 'a'];
        }
        trie[root].cnt += cnt;
    }
	// build Fail
    void build() {
        trie[0].fail = -1;
        queue<int> q;
        for(int i = 0; i < 26; ++i) {
            if(trie[0].net[i]) {
                trie[trie[0].net[i]].fail = 0;
                q.push(trie[0].net[i]);
            }
        }
        while (!q.empty()) {
            int now = q.front();
            q.pop();
            for (int i = 0; i < 26; ++i) {
                int Fail = trie[now].fail;
                int to = trie[now].net[i];
                if (to) {
                    trie[to].fail = trie[Fail].net[i];
                    q.push(to);
                } else trie[now].net[i] = trie[Fail].net[i];
            }
        }
    }
	// jump Fail
    int query(const string &s) {
        int root = 0, ans = 0;
        for(int i = 0, len = s.length(); i < len; ++i) {
            int id = s[i] - 'a';
            int k = trie[root].net[id];
            while(k && trie[k].cnt != -1) {
                ans += trie[k].cnt;
                trie[k].cnt = -1;
                k = trie[k].fail;
            }
            root = trie[root].net[id];
        }
        return ans;
    }
} aho;


//计算每个模式串出现的次数：
struct ACAM {
    static const int maxn = 2e5 + 100;
    struct node {
        int net[26];
        int fail, id;
    }trie[maxn];
    int tot = 0;
    int degree[maxn], sum[maxn], ans[maxn];
    void init() {
        memset(degree, 0, sizeof(degree));
        memset(sum, 0, sizeof(sum));
        memset(ans, 0, sizeof(ans));
        for(int i = 0; i < maxn; ++i) {
            memset(trie[i].net, 0, sizeof(trie[i].net));
            trie[i].id = trie[i].fail = 0;
        }
        tot = 0;
    }
    void insert(const string &s, int num) {
        int root = 0;
        for(int i = 0, len = s.length(); i < len; ++i) {
            int id = s[i] - 'a';
            if(!trie[root].net[id]) trie[root].net[id] = ++tot;
            root = trie[root].net[id];
        }
        trie[root].id = num;
    }
    void build() {
        queue<int> q;
        for(int i = 0; i < 26; ++i) {
            if(trie[0].net[i]) {
                trie[trie[0].net[i]].fail = 0;
                q.push(trie[0].net[i]);
            }
        }
        trie[0].fail = -1;
        while(!q.empty()) {
            int now = q.front(); q.pop();
            for(int i = 0; i < 26; ++i) {
                int to = trie[now].net[i];
                int Fail = trie[now].fail;
                if(to) {
                    trie[to].fail = trie[Fail].net[i];
                    degree[trie[to].fail]++;////易错
                    q.push(to);
                } else trie[now].net[i] = trie[Fail].net[i];
            }
        }
    }
	// Fail is a DAG 
    void query(const string &s) {
        int root = 0;
        for(int i = 0, len = s.length(); i < len; ++i) {
            int id = s[i] - 'a';
            root = trie[root].net[id];
            sum[root]++;
        }
    }
   // topo and calc ans
    void topo() {
        queue<int> q;
        for(int i = 1; i <= tot; ++i) {
            if(degree[i] == 0) q.push(i);
        }
        while(!q.empty()) {
            int now = q.front(); q.pop(); ans[trie[now].id] = sum[now];
            int fa = trie[now].fail; degree[fa]--;
            sum[fa] += sum[now];
            if(!degree[fa]) q.push(fa);
        }
    }
} aho;

//计算出现最多次数的子串
struct ACAM {
    static const int maxn = 1e6 + 100;
    int tot = 0;
    struct node {
        int net[26];
        int fail, id;
    }trie[maxn];
    void init() {
        for(int i = 0; i < maxn; ++i) {
            memset(trie[i].net, 0, sizeof(trie[i].net));
            trie[i].id = trie[i].fail = 0;
        }
        tot = 0;
    }
    void insert(const string &s, int cnt) {
        int root = 0;
        for(int i = 0, len = s.length(); i < len; ++i) {
            if(!trie[root].net[s[i] - 'a']) {
                trie[root].net[s[i] - 'a'] = ++tot;
            }
            root = trie[root].net[s[i] - 'a'];
        }
        trie[root].id = cnt;
    }
    void build() {
        trie[0].fail = -1;
        queue<int> q;
        for(int i = 0; i < 26; ++i) {
            if(trie[0].net[i]) {
                trie[trie[0].net[i]].fail = 0;
                q.push(trie[0].net[i]);
            }
        }
        while (!q.empty()) {
            int now = q.front();
            q.pop();
            for (int i = 0; i < 26; ++i) {
                int Fail = trie[now].fail;
                int to = trie[now].net[i];
                if (to) {
                    trie[to].fail = trie[Fail].net[i];
                    q.push(to);
                } else trie[now].net[i] = trie[Fail].net[i];
            }
        }
    }
    void query(const string &s) {
        int root = 0;
        for(int i = 0, len = s.length(); i < len; ++i) {
            int id = s[i] - 'a';
            int k = trie[root].net[id];
			// jump Fail
            while(k) {
                if(trie[k].id) {
                    ans[trie[k].id] ++;
                    MAX = max(ans[trie[k].id], MAX);
                }
                k = trie[k].fail;
            }
            root = trie[root].net[id];
        }
    }
} aho;
```

### 回文自动机

```cpp
char s[maxn];		//原串
int fail[maxn];		//fail指针
int len[maxn];		//该节点表示的字符串长度
int tree[maxn][26];	//同Trie，指向儿子
int trans[maxn];	//trans指针
int tot,pre;		//tot代表节点数，pre代表上次插入字符后指向的回文树位置
int getfail(int x,int i){		//从x开始跳fail，满足字符s[i]的节点
	while(i-len[x]-1<0||s[i-len[x]-1]!=s[i])x=fail[x];
	return x;
}
int gettrans(int x,int i){
	while(((len[x]+2)<<1)>len[tot]||s[i-len[x]-1]!=s[i])x=fail[x];
	return x;
}
void insert(int u,int i){
	int Fail=getfail(pre,i);		//找到符合要求的点
	if(!tree[Fail][u]){		//没建过就新建节点
		len[++tot]=len[Fail]+2;	//长度自然是父亲长度+2
		fail[tot]=tree[getfail(fail[Fail],i)][u];	//fail为满足条件的次短回文串+u
		tree[Fail][u]=tot;		//指儿子
		if(len[tot]<=2)trans[tot]=fail[tot];	//特殊trans
		else{
			int Trans=gettrans(trans[Fail],i);	//求trans
			trans[tot]=tree[Trans][u];
		}
	}
	pre=tree[Fail][u];		//更新pre
}
```

### 后缀数组

```cpp
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
using namespace std;

const int N = 200005;

int n;
char s[N];

namespace SA {
int sa[N], rk[N], ht[N], s[N<<1], t[N<<1], p[N], cnt[N], cur[N], mi[N][25];
#define pushS(x) sa[cur[s[x]]--] = x
#define pushL(x) sa[cur[s[x]]++] = x
#define inducedSort(v) 
    fill_n(sa, n, -1); fill_n(cnt, m, 0);                               
    for (int i = 0; i < n; i++) cnt[s[i]]++;                             
    for (int i = 1; i < m; i++) cnt[i] += cnt[i-1];                     
    for (int i = 0; i < m; i++) cur[i] = cnt[i]-1;                       
    for (int i = n1-1; ~i; i--) pushS(v[i]);                             
    for (int i = 1; i < m; i++) cur[i] = cnt[i-1];                       
    for (int i = 0; i < n; i++) if (sa[i] > 0 &&  t[sa[i]-1]) pushL(sa[i]-1); 
    for (int i = 0; i < m; i++) cur[i] = cnt[i]-1;                       
    for (int i = n-1;  ~i; i--) if (sa[i] > 0 && !t[sa[i]-1]) pushS(sa[i]-1);
void sais(int n, int m, int *s, int *t, int *p) {
    int n1 = t[n-1] = 0, ch = rk[0] = -1, *s1 = s+n;
    for (int i = n-2; ~i; i--) t[i] = s[i] == s[i+1] ? t[i+1] : s[i] > s[i+1];
    for (int i = 1; i < n; i++) rk[i] = t[i-1] && !t[i] ? (p[n1] = i, n1++) : -1;
    inducedSort(p);
    for (int i = 0, x, y; i < n; i++) if (~(x = rk[sa[i]])) {
        if (ch < 1 || p[x+1] - p[x] != p[y+1] - p[y]) ch++;
        else for (int j = p[x], k = p[y]; j <= p[x+1]; j++, k++)
            if ((s[j]<<1|t[j]) != (s[k]<<1|t[k])) {ch++; break;}
        s1[y = x] = ch;
    }
    if (ch+1 < n1) sais(n1, ch+1, s1, t+n, p+n1);
    else for (int i = 0; i < n1; i++) sa[s1[i]] = i;
    for (int i = 0; i < n1; i++) s1[i] = p[sa[i]];
    inducedSort(s1);
}
template<typename T>
int mapCharToInt(int n, const T *str) {
    int m = *max_element(str, str+n);
    fill_n(rk, m+1, 0);
    for (int i = 0; i < n; i++) rk[str[i]] = 1;
    for (int i = 0; i < m; i++) rk[i+1] += rk[i];
    for (int i = 0; i < n; i++) s[i] = rk[str[i]] - 1;
    return rk[m];
}

template<typename T>
void suffixArray(int n, const T *str) {
    int m = mapCharToInt(++n, str);
    sais(n, m, s, t, p);
    for (int i = 0; i < n; i++) rk[sa[i]] = i;
    for (int i = 0, h = ht[0] = 0; i < n-1; i++) {
        int j = sa[rk[i]-1];
        while (i+h < n && j+h < n && s[i+h] == s[j+h]) h++;
        if (ht[rk[i]] = h) h--;
    }
}

void RMQ_init() {
	for(int i = 0; i < n; ++i) mi[i][0] = ht[i + 1];
	for(int j = 1; (1 << j) <= n; ++j){
		for(int i = 0; i + ( 1 << j) <= n; ++i){
			mi[i][j] = min(mi[i][j - 1], mi[i + (1 << (j - 1))][j - 1]);
		}
	}
}

int RMQ(int L, int R) {
	int k = 0, len = R - L + 1;
	while( ( 1 << (k + 1)) <= len) ++k;
	return min(mi[L][k], mi[R - (1 << k) + 1][k]);
}

int LCP(int i, int j) {
	if(rk[i] > rk[j]) swap(i, j);
	return RMQ(rk[i], rk[j] - 1);
}

template<typename T>
void init(T *str){
	n = strlen(str);
	str[n] = 0;
	suffixArray(n, str);
	RMQ_init();
}
};

//读入从0开始 
int main()
{
	scanf("%s", s);
	n = strlen(s);
	s[n] = 'a' - 1;

	SA::suffixArray(n, s);

	for (int i = 1; i <= n; ++i)
		printf("%d ", SA::sa[i] + 1);
	printf("\n");
	for (int i = 2; i <= n; ++i)
		printf("%d ", SA::ht[i]);

	return 0;
}
```

## 数学

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

## 图论

### 点分治

处理树上路径问题

基本思路：

处理经过当前根节点的路径

删除当前节点分治处理子树

为保证复杂度 选择重心为根节点

```cpp
int head[N], tot = 0;

void add(int x, int y, int w) {
    s[++tot] = {y, head[x], w};
    head[x] = tot;
}

int root, k;

int siz[N], vis[N];

void dfs_rt(int node, int fa, int tot) { // find root
    siz[node] = 1;
    int maxn = 0;
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(to == fa or vis[to]) continue;
        dfs_rt(to, node, tot);
        siz[node] += siz[to];
        if(siz[to] > maxn) maxn = siz[to];
    }
    maxn = max(maxn, tot - siz[node]);
    if(maxn * 2 <= tot) root = node;
}

void dfs_clear(int node, int fa, int dis) { // del data
    if(dis) tree.add(dis, -1);
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        int w = s[i].w;
        if(to == fa or vis[to]) continue;
        dfs_clear(to, node, dis + w);
    }
}

int d[N], cnt = 0;

void dfs_dis(int node, int fa, int dis) { // calc dis
    d[++cnt] = dis;
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        int w = s[i].w;
        if(to == fa or vis[to]) continue;
        dfs_dis(to, node, dis + w);
    }
}


int divide(int node, int tot) {
    dfs_rt(node, 0, tot);
    node = root;
    dfs_rt(node, 0, tot);

    int ans = 0;
    vis[node] = 1;

    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        int w = s[i].w;
        if(vis[to]) continue;

        cnt = 0;
        dfs_dis(to, node, w);

        for(int i = 1; i <= cnt; ++i) {
            if(d[i] <= k) ++ans;
            ans += tree.ask(max(0, k - d[i]));
        }
        for(int i = 1; i <= cnt; ++i) {
            if(d[i] <= k) {
                tree.add(d[i], 1);
            }
        }
    }

    dfs_clear(node, 0, 0);

    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(vis[to]) continue;

        ans += divide(to, siz[to]);

    }
    return ans;
}
```

### 次小生成树

```cpp
const int maxn = 1000 + 10, maxe = 1000 * 1000 / 2 + 5, INF = 0x3f3f3f3f;
int n, m, pre[maxn], head[maxn], Max[maxn][maxn];
struct Edge {
    int u, v, w;
    bool vis;
    bool operator < (const Edge &rhs) const{
        return w < rhs.w;
    }
}edge[maxe];
vector<int> G[maxn];

void Init() {
    for(int i = 1; i <= n; i ++) {
        G[i].clear();
        G[i].push_back(i);
        head[i] = i;
    }
}

int Find(int x) {
    if(head[x] == x) return x;
    return head[x] = Find(head[x]);
}

int Kruskal() {
    sort(edge + 1, edge + 1 + m);
    Init();
    int ans = 0, cnt = 0;
    for(int i = 1; i <= m; i ++) {
        if(cnt == n - 1) break;
        int fx = Find(edge[i].u), fy = Find(edge[i].v);
        if(fx != fy) {
            cnt ++;
            edge[i].vis = true;
            ans += edge[i].w;
            int len_fx = G[fx].size(), len_fy = G[fy].size();
            for(int j = 0; j < len_fx; j ++)
                for(int k = 0; k < len_fy; k ++)
                    Max[G[fx][j]][G[fy][k]] = Max[G[fy][k]][G[fx][j]] = edge[i].w;
            head[fx] = fy;
            for(int j = 0; j < len_fx; j ++)
                G[fy].push_back(G[fx][j]);
        }
    }
    return ans;
}

int Second_Kruskal(int MST) {
    int ans = INF;
    for(int i = 1; i <= m; i ++)
        if(!edge[i].vis)
            ans = min(ans, MST + edge[i].w - Max[edge[i].u][edge[i].v]);
    return ans;
}

int main() {
    scanf("%d %d", &n, &m);
    for(int i = 1; i <= m; i ++) {
        scanf("%d %d %d", &edge[i].u, &edge[i].v, &edge[i].w);
        edge[i].vis = false;
    }
    int MST = Kruskal();
    int Second_MST = Second_Kruskal(MST);
    printf("%d\n", Second_MST );
}
```

### 树上倍增求链上最大值

```cpp
void predfs(int node, int fa) {
    dep[node] = dep[fa] + 1;
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(to == fa) {
            maxv[0][node] = s[i].w;
            continue;
        }
        predfs(to, node);
    }
}

void dfs(int node, int fa) {
    dp[0][node] = fa;
    for(int i = 1; (1 << i) <= dep[node]; ++i) {
        dp[i][node] = dp[i - 1][dp[i - 1][node]];
        maxv[i][node] = max(maxv[i - 1][node], maxv[i - 1][dp[i - 1][node]]);
    }
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(to != fa) dfs(to, node);
    }
}

int LCA(int x, int y) {
    int maxx = 0;
    if(dep[x] > dep[y]) swap(x, y);
    int k = log2(dep[y]);
    for(int i = k; i >= 0; --i) {
        if(dep[dp[i][y]] >= dep[x]) {
            maxx = max(maxx, maxv[i][y]);
            y = dp[i][y];
        }
    }
    if(x == y) return maxx;
    for(int i = k; i >= 0; --i) {
        if(dp[i][y] != dp[i][x]) {
            maxx = max(maxx, maxv[i][y]);
            maxx = max(maxx, maxv[i][x]);
            y = dp[i][y];
            x = dp[i][x];
        }
    }
    return max(maxx, max(maxv[0][x], maxv[0][y]));
}
```

### 树上差分

```cpp
vector<int> v[N];

int cfone[N], cftwo[N];
int fa[N], dep[N], dp[N][25];

void predfs(int node, int f) {
    dep[node] = dep[f] + 1;
    dp[node][0] = f;
    fa[node] = f;
    for(int i = 1; (1 << i) <= dep[node]; ++i) {
        dp[node][i] = dp[dp[node][i - 1]][i - 1];
    }
    for(auto k : v[node]) {
        if(k == f) continue;
        predfs(k, node);
    }
}

int Lca(int x, int y) {
    if(dep[x] < dep[y]) swap(x, y);
    int tem = dep[x] - dep[y];
    for(int i = 0; tem; ++i) {
        if(tem & 1) x = dp[x][i];
        tem >>= 1;
    }
    if(x == y) return x;
    for(int j = 22; j >= 0 && x != y; --j) {
        if(dp[x][j] != dp[y][j]) {
            x = dp[x][j];
            y = dp[y][j];
        }
    }
    return dp[x][0];
}

void dfs(int node, int fa) {
    for(auto to : v[node]) {
        if(to == fa) continue;
        dfs(to, node);
        cfone[node] += cfone[to];
        cftwo[node] += cftwo[to];
    }
}

int main() {
    predfs(1, 0);
    int k = gn();
    int pre = 1;
    for(int i = 1; i <= k; ++i) {
        int now = gn();
        int lca = Lca(pre, now);
        cfone[pre]++, cfone[lca]--;
        cftwo[now]++, cftwo[lca]--;
        // cf[u]++, cf[v]++, cf[lca(u, v)]--, cf[fa(lca(u, v))]--;
        pre = now;
    }
    ll ans = 0;
    dfs(1, 0);
}
```

### 01MST

```cpp
constexpr int mod = 1e9 + 7;
constexpr int N = 2e5 + 100;

set<int> st, mp[N];
int vis[N];

int main() {
    int n = gn(), m = gn();
    for(int i = 1; i <= n; ++i) {
        st.insert(i);
    }
    for(int i = 1; i <= m; ++i) {
        int x = gn(), y = gn();
        mp[x].insert(y);
        mp[y].insert(x);
    }
    queue<int> q;
    int ans = 0;
    for(int i = 1; i <= n; ++i) {
        if (vis[i]) continue;
        q.push(i);
        ans ++;
        st.erase(i);
        while(!q.empty()) {
            int now = q.front(); q.pop();
            if (vis[now]) continue;
            vis[now] = 1;
            for(auto k = st.begin() ; k != st.end(); ) {
                if( !mp[now].count(*k) ) {
                    q.push(*k);
                    k = st.erase(k);
                } else ++k;
            }
        }
    }
    printf("%d\n", ans - 1);
}
```

### 矩阵树定理

```cpp
#define MAXN 20
#define eps 1e-9
using namespace std;
int A[MAXN][MAXN], D[MAXN][MAXN];//A是邻接矩阵D是度数矩阵 
double C[MAXN][MAXN];//基尔霍夫矩阵 
int T, n, m;
double gauss() {
    int now = 1;
    for (int i = 1; i <= n; ++i) {
        int x = now;
        while (fabs(C[x][now]) < eps && x <= n) x++;
        if (x == n + 1) {
            return 0;
        }
        for (int j = 1; j <= n; ++j)  swap(C[now][j],C[x][j]);
        for (int j = now + 1; j <= n; ++j) {
            double temp = C[j][now] / C[now][now];
            for (int k = 1;k <= n;k++)
                C[j][k] -= temp * C[now][k];
        }
        now++;
    }
    double ans=1;
    for (int i = 1; i <= n; ++i) ans *= C[i][i];
    ans = fabs(ans);
    return ans;
}
int main(){
    memset(A, 0, sizeof(A));
    memset(D, 0, sizeof(D));
    for (int i = 1; i <= m; ++i)
    {
        int u, v;
        D[u][u]++;D[v][v]++;
        A[u][v]++;A[v][u]++;
    }
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
            C[i][j] = D[i][j] - A[i][j];
    gauss();
}
```



### tarjan求点双连通分量

```jsx
int dfn[N], low[N], vis[N];
int tim = 0, colornum = 0;
stack<int> q;

int siz[N], colornum = 0;

void tarjan(int u, int fa) {
    dfn[u] = low[u] = ++tim;
    vis[u] = 1;
    q.push(u);
    for(int i = head[u]; ~i; i = s[i].net) {
        int to = s[i].to;
        if (to == fa) continue;
        if(!dfn[to]) {
            int z;
            tarjan(to, u);
            low[u] = min(low[to], low[u]);
            if(low[to] >= dfn[u]) {
                int len = 0;
                ++colornum;
                do {
                    ++len;
                    z = q.top();
                    q.pop();
                    vis[z] = 0;
                } while(z != to);
                siz[colornum] = len + 1;
            }
        } else if (vis[u]) low[u] = min(low[u], dfn[to]);
    }

}
```

### 性质

- 在一个无权图上求从起点到其他所有点的最短路径。
- 在$O(n + m)$时间内求出所有连通块。(我们只需要从每个没有被访问过的节点开始做BFS，显然每次BFS会走完一个连通块)
- 如果把一个游戏的动作看做是状态图上的一条边(一 个转移)， 那么BFS可以用来找到在游戏中从一个状态到达另一个状态所需要的最小步骤。
- 在一个边权为0/1的图上求最短路。 (需要修改入队的过程,如果某条边权值为0，且可以减小边的终点到图的起点的距离，那么把边的起点加到队列首而不是队列尾)
- 在一个有向无权图中找最小环。 (从每个点开始BFS, 在我们即将抵达一 个之前访问过的点开始的时候，就知道遇到了一个环。图的最小环是每次BFS得到的最小环的平均值。)
- 找到一定在$(a,b)$最短路上的边。 (分别从a和b进行BFS， 得到两个d数组。之后对每一条边$(u,v)$,如果$da[u]+1 + db[v]= da[b]$ ,则说明该边在最短路上)
- 找到一定在$(a,b)$最短路上的点。 (分别从a和b进行BFS, 得到两个d数组。之后对每一个点v,如果$da[u] + db[v] = da[b]$，则说明该点在最短路上)
- 找到一条长度为偶数的最短路。 (我们需要一 个构造一 个新图,把每个点拆成两个新点,原图的边$(u,v)$变成$((u, 0), (v,1))$和$((u,1),(v,0))$。对新图做BFS，$(s, 0)$和$(t,0)$之间的最短路即为所求)
- 树上的最远距离是树的直径，树的直径可以用两次搜索写出来。树上每个节点离他最远的节点在该树的直径上
- 前序遍历中，$LCA(S)$ 出现在所有S中元素之前，后序遍历中$LCA(S)$则出现在所有S中元素之后
- 两点集并的最近公共祖先为两点集分别的最近公共祖先的最近公共祖先，即$LCA(A∪B) = LCA(LCA(A), LCA(B))$
- 两点的最近公共祖先必定处在树上两点间的最短路上
$d(u,v)= h(u) + h(v) - 2h(LCA(u,v))$,其中d是树上两点间的距离，h代表某点到树根的距离
- 以树的重心为根时，所有子树的大小都不超过整棵树大小的一半
- 树中所有点到某个点的距离和中，到重心的距离和是最小的;如果有两个重心,那么到它们的距离和一样
- 把两棵树通过一条边相连得到一棵新的树， 那么新的树的重心在连接原来两棵树的重心的路径上。
- 在一棵树上添加或删除一 个叶子，那么它的重心最多只移动一条边的距离
- 计算(u,v)边的最短路经过次数 $sz[v]∗(n−sz[v])$,
- 计算经过某点的最短路次数: 计算和这个点相连的所有边的被计算次数加上$n−1$(这个点自身和其他所有点)再除以2

### 次短路

```cpp
struct edge{
    int to, cost;
    edge(int tv = 0, int tc = 0):
            to(tv), cost(tc){}
};
typedef pair<int ,int> P;
int N, R;
vector<edge> graph[MAXN];
int dist[MAXN];     //最短距离
int dist2[MAXN];    //次短距离

void solve() {
    fill(dist, dist + N, INF);
    fill(dist2, dist2 + N, INF);
    priority_queue <P, vector<P>, greater<P>> Q;
    dist[0] = 0;
    Q.push(P(0, 0));
    while (!Q.empty()) {
        P p = Q.top();
        Q.pop();
        //first为s->to的距离，second为edge结构体的to
        int v = p.second, d = p.first;
        //当取出的值不是当前最短距离或次短距离，就舍弃他
        if (dist2[v] < d) continue;
        for (unsigned i = 0; i < graph[v].size(); i++) {
            edge &e = graph[v][i];
            int d2 = d + e.cost;
            if (dist[e.to] > d2) {
                swap(dist[e.to], d2);
                Q.push(P(dist[e.to], e.to));
            }
            if (dist2[e.to] > d2 && dist[v] < d2) {
                dist2[e.to] = d2;
                Q.push(P(dist2[e.to], e.to));
            }
        }
    }
    printf("%d\n", dist2[N - 1]);
}
```

### Johnson 全源最短路(适用于稀疏图）

```cpp
struct node{
    int to, net;
    ll w;
}s[N << 2];

int tot = 0, head[N], n, m, S;

ll dis[N], h[N];

int cnt[N], vis[N];

void add(int x, int y, ll w) {
    s[++tot] = {y, head[x], w};
    head[x] = tot;
}

ll dij(ll start) {
    priority_queue<pair<ll, int> > q;
    for(int i = 1; i <= n; ++i) h[i] = 1e9;
    h[start] = 0;
    q.push({0, start});
    while(!q.empty()) {
        int now = q.top().second;
        ll distance = -q.top().first;
        q.pop();
        if(distance > h[now]) continue;
        for(int i = head[now]; ~i; i = s[i].net) {
            int to = s[i].to;
            if(h[to] > h[now] + s[i].w) {
                h[to] = h[now] + s[i].w;
                q.push({-h[to], to});
            }
        }
    }
    ll ans = 0;
    for(int i = 1; i <= n; ++i) {
        if(i == start) continue;
        if(h[i] == 1e9) ans += i * 1e9;
        else ans += i * (h[i] + dis[i] - dis[start]);
    }
    return ans;
}

bool spfa() {
    memset(dis, 0x3f, sizeof(dis));
    deque<int> q;
    dis[S] = 0;
    vis[S] = 1;
    cnt[S] = 0;
    q.push_front(S);
    while(!q.empty()) {
        int now = q.front();
        q.pop_front();
        vis[now] = 0;
        if(cnt[now] > n) return false;
        cnt[now] ++;
        for(int i = head[now]; ~i; i = s[i].net) {
            int to = s[i].to;
            if(dis[to] > dis[now] + s[i].w) {
                dis[to] = dis[now] + s[i].w;
                if(!vis[to]) {
                    if(!q.empty() || dis[to] < dis[q.front()]) q.push_front(to);
                    else q.push_back(to);
                    vis[to] = 1;
                }
            }
        }
    }
    return true;
}

int main() {
    memset(head, -1, sizeof(head));
    n = gn(), m = gn();
    for(int i = 1; i <= m; ++i) {
        int x = gn(), y = gn();
        ll w = gl();
        add(x, y, w);
    }
    S = 0;
    for(int i = 1; i <= n; ++i) {
        add(S, i, 0LL);
    }
    if(!spfa()) {
        cout << -1 << endl;
    } else {
        for(int i = 1; i <= n; ++i) {
            for(int j = head[i]; ~j; j = s[j].net) {
                s[j].w += dis[i] - dis[ s[j].to ];
            }
        }
        
        for(ll i = 1; i <= n; ++i) {
            cout << dij(i) << endl;
        }
    }
}
```

### 2-SAT 缩点解法

```jsx
struct node{
    int to,net,w;
}s[N << 1];
int head[N], ans[N], tot = 0;
void add(int a, int b, int w){
    s[++tot] = {b, head[a], w};
    head[a] = tot;
}
int n, m;
stack<int> q;
int dfn[N], low[N], color[N], colornum=0, tim=0, vis[N];
void targin(int node) {
    dfn[node] = low[node] = ++tim;
    vis[node] = 1;
    q.push(node);
    for(int i = head[node]; ~i; i = s[i].net) {
        if(!dfn[s[i].to]) {
            targin(s[i].to);
            low[node] = min(low[node], low[s[i].to]);
        }else if(vis[s[i].to]) {
            low[node] = min(low[node],dfn[s[i].to]);
        }
    }
    if(low[node]==dfn[node]) {
        ++colornum;
        int z;
        do{
            z = q.top();
            q.pop();
            color[z] = colornum;
            vis[z] = 0;
        }while(z != node);
    }
}
int main() {
    memset(head,-1,sizeof(head));
    n = gn(),m = gn();
    for(int i = 1; i <= m; ++i) {
        int a = gn(), vala = gn(), b = gn(), valb = gn();
        add(a + vala * n, b + (1 - valb) * n, 1);
        add(b + valb * n, a + (1 - vala) * n, 1);
    }
    for(int i = 1; i <= n + n; ++i){
        if(!dfn[i]) {
            targin(i);
        }
    }
    for(int i = 1; i <= n; ++i) {
        if(color[i] == color[i+n]){
            puts("IMPOSSIBLE");
            return 0;
        }else {
            if(color[i] < color[i+n]){
                ans[i] = 1;
            }else ans[i] = 0;
        }
    }
    puts("POSSIBLE");
    for(int i = 1; i <= n; ++i) {
        putchar(ans[i] + '0'); putchar(' ');
    }
}
```

### 费用流

```jsx
int n, m, S, T, tot = -1;
int head[N], dis[N], vis[N], pre[N];
ll max_flow = 0, min_cost = 0, incf[N];
struct node {
    int to, net;
    ll w;
    int c;
}s[N];
void add(int x, int y, int w, int c) {
    s[++tot] = {y, head[x], w, c};
    head[x] = tot;
    s[++tot] = {x, head[y], 0, -c};
    head[y] = tot;
}
bool spfa() {
    memset(dis, 0x3f, sizeof(dis));
    memset(vis, 0, sizeof(vis));
    dis[S] = 0;
    queue <int> q;
    q.push(S);
    vis[S] = 1;
    incf[S] = INF;
    pre[T] = 0;
    while(!q.empty()) {
        int now = q.front();
        q.pop();
        vis[now] = 0;
        for(int i = head[now]; ~i; i = s[i].net) {
            int to = s[i].to;
            if(dis[to] > dis[now] + s[i].c && s[i].w) {
                dis[to] = dis[now] + s[i].c;
                incf[to] = min(incf[now], s[i].w);
                pre[to] = i;
                if(!vis[to]) {
                    q.push(to);
                    vis[to] = 1;
                }
            }
        }
    }
    if(dis[T] == 0x3f3f3f3f) return false;
    int x = T;
    while(x != S){
        int i = pre[x];
        s[i].w -= incf[T];
        s[i ^ 1].w += incf[T];
        x = s[i ^ 1].to;

    }
    max_flow += incf[T];
    min_cost += incf[T] * dis[T];
    return true;
}
int main(){
    memset(head, -1, sizeof(head));
    n = gn(), m = gn(), S = gn(), T = gn();
    repi(i, 1, m) {
        int x = gn(), y = gn(), w = gn(), c = gn();
        add(x, y, w, c);
    }
    while(spfa());
    print(max_flow), putchar(' '), print(min_cost);
}
```

### 树的直径(DP)

```jsx
void dfs(int node,int fa){
    f[node] = h[node];
    ll maxn = 0,minx = 0;
    for(int k : v[node]){
        if(k == fa)continue;
        dfs(k, node);
        if(f[k] > maxn) minx = maxn, maxn = f[k];
        else if(f[k] > minx) minx = f[k];
    }
    f[node] += maxn;
    ans = max(ans, maxn + minx + h[node]);
}
```

### 启发式合并

```cpp
//用于离线解决子树的一系列问题
const int N = 1e5 + 100;
const int mod = 1e9 + 7;
struct node{
    int to, net;
}s[N << 1];
int head[N], tot = 0;
int siz[N], son[N], c[N], nowson, cnt[N];
ll sum, maxn, ans[N];
 
inline void add(int a,int b){
    s[++tot] = {b, head[a]};
    head[a] = tot;
    s[++tot] = {a, head[b]};
    head[b] = tot;
}
 
inline void dfs(int node, int fa){
    siz[node] = 1;
    int maxn = 0;
    for(register int i = head[node]; ~i; i = s[i].net){
        int k = s[i].to;
        if(k == fa)continue;
        dfs(k, node);
        siz[node] += siz[k];
        if(siz[k] > maxn){
            maxn = siz[k];
            son[node] = k;
        }
    }
}
//统计节点本身和他的轻儿子的贡献
inline void cal(int node, int fa, int val){
    cnt[c[node]] += val;
    if(cnt[c[node]] > sum){
        sum = cnt[c[node]];
        maxn = c[node];
    }else if(cnt[c[node]] == sum){
        maxn += c[node];
    }
    for(register int i = head[node]; ~i; i=s[i].net) {
        int k = s[i].to;
        if(k == fa || k == nowson)continue;
        cal(k, node, val);
    }
}
 
inline void dsu(int node,int fa,bool keep){
    for(register int i = head[node]; ~i; i = s[i].net) {
        int k = s[i].to;
        if(k == fa || k == son[node]) continue;
        dsu(k, node, false);
    }
    if(son[node]) {
        dsu(son[node], node, true);
        nowson = son[node];
    }
    cal(node, fa, 1);
    nowson = 0;
    ans[node] = maxn;
    if(!keep) {
        cal(node, fa, -1);
        sum = maxn = 0;
    }
}


// version two
void predfs(int node, int fa) {
    int maxn = 0;
    siz[node] = 1;
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(to == fa) continue;
        predfs(to, node);
        siz[node] += siz[to];
        if(siz[to] > maxn) {
            maxn = siz[to];
            son[node] = to;
        }
    }
}

void dfs1(int node, int fa, int val) {
    num[c[node]] += val;
    if(num[c[node]] > maxn) {
        maxn = num[c[node]];
        ANS = c[node];
    } else if(num[c[node]] == maxn) {
        ANS += c[node];
    }
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(to == fa) continue;
        dfs1(to, node, val);
    }
}

void dsu(int node, int fa, bool sign) {
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(to == fa || to == son[node]) continue;
        dsu(to, node, false);
    }

    if(son[node]) dsu(son[node], node, true);

    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(to == fa || to == son[node]) continue;
        dfs1(to, node, 1);
    }

    if(++num[c[node]] > maxn) {
        ANS = c[node];
        maxn = num[c[node]];
    } else if(num[c[node]] == maxn) {
        ANS += c[node];
    }

    ans[node] = ANS;

    if(!sign) {
        dfs1(node, fa, -1);
        maxn = 0; ANS = 0;
    }
}
```

### LCA

```cpp
void dfs(int node,int fa) {
    dep[node] = dep[fa] + 1;
    dp[node][0] = fa;
    for(int i = 1; (1 << i) <= dep[node]; ++i) {
        dp[node][i] = dp[dp[node][i-1]][i-1];
    }
    for(int i = head[node]; ~i; i = s[i].net) {
        if(s[i].to == fa) continue;
        dfs(s[i].to, node);
    }
}
int lca(int x, int y) {
    if(dep[x] < dep[y]) swap(x, y);
    int tep = dep[x] - dep[y];
    for(int j = 0; tep; ++j) {
        if(tep & 1) x = dp[x][j];
        tep >>= 1;
    }
    if(x == y)return x;
    for(int j = 21; j >= 0 && x != y; --j) {
        if(dp[x][j] != dp[y][j]) {
            x = dp[x][j];
            y = dp[y][j];
        }
    }
    return dp[x][0];
}

//树链剖分求法
vector<int> v[N];
int dep[N], siz[N], top[N], son[N], b[N], f[N];
void dfs(int node, int fa) {
    f[node] = fa;
    siz[node] = 1;
    dep[node] = dep[fa] + 1;
    int maxn = 0;
    for(int k : v[node]) {
        if(k == fa)continue;
        dfs(k, node);
        siz[node] += siz[k];
        if(siz[k] > maxn) {
            son[node] = k;
            maxn = siz[k];
        }
    }
}
void dfs1(int node, int topx){
    top[node] = topx;
    if(son[node]) {
        dfs1(son[node], topx);
    }
    for(int k : v[node]){
        if(k == f[node] || k == son[node]) continue;
        dfs1(k, k);
    }
}
int lca(int x, int y){
    while(top[x] != top[y]){
        if(dep[top[x]] >= dep[top[y]]){
            x = f[top[x]];
        }else y = f[top[y]];
    }
    return dep[x] < dep[y] ? x : y;
}
```

### 缩点

(缩完后可以跑dp)

```cpp
/*
题意：
给定一个n个点m条边有向图, 每个点有一个权值, 求一条路径,使路径经过的点权值之和最大。 你只需要求出这个权值和。 允许多次经过一条边或者一个点, 但是, 重复经过的点, 权值只计算一次。

思路：
一个强连通分量内的点可以互相到达, 所以我们只需要缩点之后进行dp求一个最大的权值和就好
*/
#define ll long long
const int N = 1e4 + 100;
const int mod = 1e9 + 7;

vector<int> v[N], e[N]; // v存旧图 e存新图
int a[N], dfn[N], low[N], vis[N], color[N];
int colornum = 0, n, m, tim = 0;
ll k[N], dp[N], ans;
stack<int> q;

namespace shrink_point{
    //tarjan求SCC
    void tarjan(int x) {
        dfn[x] = low[x] = ++tim;
        vis[x] = 1;
        q.push(x);
        for(int i = 0,len = v[x].size(); i < len; ++i) {
            int y = v[x][i];
            if(!dfn[y]) {
                tarjan(y);
                low[x] = min(low[x], low[y]);
            } else if(vis[y]) {
                low[x] = min(low[x], dfn[y]);
            }
        }
        if(low[x] == dfn[x]) {
            ++colornum;
            int z;
            do {
                z = q.top();
                q.pop();
                color[z] = colornum;
                k[colornum] += a[z];
                vis[z] = 0;
            } while(z != x);
        }
    }
    //建立新图
    void newgraph() {
        for(int i = 1; i <= n; ++i) {
            int x = color[i];
            for(int j = 0, len = v[i].size(); j < len; ++j) {
                int y = color[ v[i][j] ];
                if(x == y) continue;
                e[y].push_back(x);
            }
        }
    }
    ll dfs(int x) {
        if(dp[x]) return dp[x];
        dp[x] = k[x];
        vis[x] = -1;
        ll sum = dp[x];
        for(int i=0,len = e[x].size();i < len; ++i){
            int y = e[x][i];
            sum=max(sum, dfs(y) + k[x]);
        }
        vis[x] = 1;
        return dp[x] = sum;
    }
    void solve() {
        n = gn(), m = gn();
        for(int i = 1; i <= n; ++i) a[i]=gn();
        for(int i = 0; i < m; ++i) {
            int x = gn(), y = gn();
            v[x].push_back(y);
        }
        for(int i = 1; i <= n; ++i) {
            if(!dfn[i]) {
                tarjan(i);
            }
        }
        newgraph();
        memset(vis, 0, sizeof(vis));
        ans = 0;
        for(int i = 1; i <= colornum; ++i){
            if(!vis[i]){
                dfs(i);
            }
        }
        for(int i = 1;i <= colornum; ++i){
            ans = max(ans, dp[i]);
        }
        print(ans);
    }

}
int main(){
    shrink_point::solve();
}
```

### tarjan求桥

```cpp
const int MAXN = 1e5 + 10;

struct node{
    int v, next, use;
}edge[MAXN << 2];

bool bridge[MAXN];
int low[MAXN], dfn[MAXN], vis[MAXN];
int head[MAXN], pre[MAXN], ip, sol, count;

void init(void){
    memset(head, -1, sizeof(head));
    memset(vis, false, sizeof(vis));
    memset(bridge, false, sizeof(bridge));
    count = sol = ip = 0;
}

void addedge(int u, int v){
    edge[ip].v = v;
    edge[ip].use = 0;
    edge[ip].next = head[u];
    head[u] = ip++;
}

void tarjan(int u) {
    vis[u] = 1;
    dfn[u] = low[u] = count++;
    for(int i = head[u]; i != -1; i = edge[i].next){
        if(!edge[i].use){
            edge[i].use = edge[i ^ 1].use = 1;
            int v = edge[i].v;
            if(!vis[v]) {
                pre[v] = u;
                tarjan(v);
                low[u] = min(low[u], low[v]);
                if(dfn[u] < low[v]){
                    sol++;
                    bridge[v] = true;
                }
            }else if(vis[v] == 1){
                low[u] = min(low[u], dfn[v]);
            }
        }
    }
    vis[u] = 2;
}

int main() {
    if(!n && !m) break;
    for(int i = 0; i < m; i++){
        scanf("%d%d", &x, &y);
        addedge(x, y); addedge(y, x);
    }
    pre[1] = 1;
    tarjan(1);
    for(int i = 1; i <= n; i++){
        if(bridge[i]) cout << i << " " << pre[i] << endl;
    }
}
```

### tarjan求强连通分量和割点

```cpp
//tarjan求强连通分量
const int N = 8500;
const int mod = 1e9 + 7;
map<string,int> mp;
vector<int> v[N];
int tot = 0, dfn[N], low[N], tim = 0, vis[N], ans[N];
stack<int> q;
void tarjan(int node) {
    dfn[node] = low[node] = ++tim;
    vis[node] = 1;
    q.push(node);
    for(int i = 0,len = v[node].size(); i < len; ++i) {
        int to = v[node][i];
        if(!dfn[to]) {
            tarjan(to);
            low[node] = min(low[to], low[node]);
        } else if(vis[to]) {
            low[node]=min(low[node], dfn[to]);
        }
    }
    map<int,bool> flag;
    int z;
    if(dfn[node] == low[node]){
        do{
            z=q.top();
            q.pop();
            if(flag[(z+1)/2]) ans[(z+1)/2]=1;
            flag[(z+1)/2]=true;
            vis[z]=0;
        }while(z!=node);
    }
}

//tarjan求割点
vector<int> v[N];
int dfn[N], low[N], tim = 0, vis[N];
set<int> ans;
void tarjan(int node, int rt){
    dfn[node] = low[node] = ++tim;
    vis[node] = 1;
    int child = 0;
    for(int i = 0,len = v[node].size(); i < len; ++i) {
        int to = v[node][i];
        if(!dfn[to]) {
            ++child;
            tarjan(to,rt);
            low[node] = min(low[to], low[node]);
            if(node != rt && low[to] >= dfn[node]) {
                ans.insert(node);
            }
        } else if(vis[to]) {
            low[node] = min(low[node], dfn[to]);
        }
    }
    if(child >= 2 && node == rt) {
        ans.insert(node);
    }
}

 for(int i = 1; i <= n; ++i) {
      if(!dfn[i]) {
          tarjan(i, i);
       }
  }
  
```

### 网络流之hlpp

(上下界卡的很紧 正常用Dinic即可

```cpp
const int N = 1250;
int head[N], h[N], gap[N], vis[N];
ll e[N];
int S,T,n,m,tot=0;
struct node {
    int to,net;
    ll w;
}s[N*200];
struct cmp{
    bool operator ()(int x,int y) {
        return h[x]<h[y];
    }
};
priority_queue<int, vector<int>, cmp> pq;
void add(int u, int v, ll w) {
    s[tot] = {v, head[u], w};
    head[u] = tot++;
    s[tot] = {u, head[v], 0LL};
    head[v] = tot++;
}
bool bfs() {
    memset(h,0x3f,sizeof(h));
    h[T] = 1;
    queue<int> q;
    q.push(T);
    while(!q.empty()) {
        int now = q.front(); q.pop();
        for(int i = head[now]; ~i; i = s[i].net) {
            if(s[i^1].w && h[s[i].to] > h[now] + 1) {
                h[s[i].to] = h[now] + 1;
                q.push(s[i].to);
                ++gap[h[s[i].to]];
            }
        }
    }
    return h[S] != 0x3f3f3f3f;
}
void push(int u){
    for(int i = head[u]; ~i && e[u]; i = s[i].net) {
        if(s[i].w && (h[s[i].to] + 1 == h[u])) {
            ll flo = min(e[u], s[i].w);
            s[i].w -= flo; s[i^1].w += flo;
            e[u] -= flo; e[s[i].to] += flo;
            if(s[i].to != S && s[i].to != T && (!vis[s[i].to])) {
                pq.push(s[i].to);
                vis[s[i].to] = 1;
            }
        }
    }
}
void relabel(int u) {
    h[u] = (1<<30);
    for(int i = head[u]; ~i; i = s[i].net){
        if(s[i].w && h[u] > h[s[i].to] + 1)h[u] = h[s[i].to] + 1;
    }
}
ll hlpp(){
    if(!bfs()) return 0;
    --gap[h[S]]; h[S]=n;
    ++gap[n];
    for(int i = head[S]; ~i; i = s[i].net) {
        if(ll flo = s[i].w) {
            s[i].w-=flo; s[i^1].w += flo;
            e[S] -= flo; e[s[i].to] += flo;
            if(s[i].to != S && s[i].to != T && !vis[s[i].to]) {
                pq.push(s[i].to);
                vis[s[i].to] = 1;
            }
        }
    }
    while(!pq.empty()) {
        int t = pq.top();
        pq.pop();
        vis[t] = 0; push(t);
        if(e[t]) {
            --gap[h[t]];
            if(!gap[h[t]]) {
                for(int j = 1; j <= n; ++j) {
                    if(j != S && j != T && h[j] > h[t]) {
                        h[j] = n + 1;
                    }
                }
            }
            relabel(t); ++gap[h[t]];
            pq.push(t), vis[t]=1;
        }
    }
    return e[T];
}
```

### 网络流之Dinic

Dinic板子：

```cpp
const int N = 12e4+100;
int n, m, S, T;
struct node {
    int to;
    ll w;
    int net;
}s[N << 1];
int head[1250], cur[1250], dep[1250],tot = 0;
void add(int u, int v, ll w){
    s[tot] = {v, w, head[u]};
    head[u] = tot++;
    s[tot] = {u, 0LL, head[v]};
    head[v] = tot++;
}
bool bfs() {
    memset(dep, 0, sizeof(dep));
    queue<int> Q;
    dep[S]=1;Q.push(S);
    while(!Q.empty()){
        int now = Q.front();
        Q.pop();
        for(int i = head[now]; ~i; i=s[i].net) {
            if(!dep[s[i].to] && s[i].w > 0) {
                dep[s[i].to] = dep[now] + 1;
                Q.push(s[i].to);
            }
        }
    }
    return dep[T];
}
ll dfs(int u, ll flo){
    if(u == T)return flo;
    ll del = 0;
    for(int i = cur[u]; (~i) && flo; i = s[i].net) {
        cur[u] = i;
        if(dep[s[i].to] == (dep[u] + 1) && s[i].w > 0) {
            int x = dfs(s[i].to, min(flo,s[i].w));
            s[i].w -= x; s[i^1].w += x; del += x; flo -= x;
        }
    }
    if(!del) dep[u] = -2;
    return del;
}
ll dinic(){
    ll ans = 0;
    while(bfs()) {
        for(int i = 1;i <= n; ++i)cur[i] = head[i];
        ans += dfs(S, (1<<30));
    }
    return ans;
}
int main(){
    n = gn(), m = gn(), S = gn(), T = gn();
    memset(head, -1, sizeof(head));
    for(int i = 0; i < m; ++i) {
        int u = gn(), v = gn();
        ll w = gl();
        add(u, v, w);
    }
    print(dinic()), putchar(10);
}
```

### 二分图

```cpp
//染色法判断二分图
bool dfs(int x, int c){
    color[x] = c;
    for(int i = 0,len = v[x].size(); i < len; ++i) {
        int m = v[x][i];
        if(!color[m]) {
            if(!dfs(m, 3 - c)) return false;
        }
        if(color[m] == c) return false;
    }
    return true;
}
bool judge(){
    for(int i = 1; i <= n; ++i) {
        if(!color[i]) {
            if(!dfs(i, 1)) return false;
        }
    }
    return true;
}

//匈牙利算法求最大匹配
bool sell(int x) {
    for(int i = 0, len = v[x].size(); ++i) {
        int m = v[x][i];
        if(!vis[m]) {
            vis[m] = 1;
            if(!fang[m] || sell(fang[m])) {
                fang[m] = x;
                return true;
            }
        }
    }
    return false;
}
for(int i = 1; i <= n1; ++i) {
    for(int j = 1; j <= n2; ++j) {
        vis[j] = 0;
    }
    if(sell(i)) ++cnt;
}

二分图中的其他性质
关于二分图中其他的性质有：
二分图的最小顶点覆盖： 用最少的点让每条边都至少和其中一个点关联。 
Knoig定理：二分图的最小顶点覆盖数等于二分图的最大匹配数。

DAG图的最小路径覆盖：用尽量少的不相交简单路径覆盖有向无环图的所有顶点。 
引理：DAG图的最小路径覆盖数=节点数(n)-最大匹配数(m)

二分图的最大独立集 在Ｎ个点的图G中选出m个点，使这m个点两两之间没有边．求m最大值。 
引理：二分图的最大独立集数 = 节点数(n)—最大匹配数(m)

//最大流求二分图匹配
const int N = 2e5 + 100;
struct node {
    int to, net, w;
}s[N * 4];
int tot = 0, head[N], S, T, dep[N], cur[N];
void add(int u, int v, int w){
    s[tot]={v, head[u], w};
    head[u] = tot++;
}
vector<pair<ll,ll> > v;
vector<ll> lt, rt;
int where(int x, bool flag){
    if(flag) return lower_bound(lt.begin(), lt.end(), x)- lt.begin() + 1;
    return lower_bound(rt.begin(), rt.end(), x) - rt.begin() + 1;
}
bool bfs() {
    memset(dep, 0, sizeof(dep));
    dep[S] = 1;
    queue<int> q;
    q.push(S);
    while(!q.empty()) {
        int now = q.front();
        q.pop();
        for(int i = head[now]; ~i; i = s[i].net) {
            int to = s[i].to;
            if(!dep[to] && s[i].w) {
                dep[to] = dep[now] + 1;
                q.push(to);
            }
        }
    }
    return dep[T];
}
int dfs(int node, int flow) {
    if(node == T) return flow;
    int del = 0;
    for(int i = cur[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        cur[node] = i;
        if(dep[to] == dep[node] + 1 && s[i].w) {
            ll w = dfs(to, min(flow - del, s[i].w));
            s[i].w -= w;
            s[i^1].w += w;
            del += w;
            if(del == flow)break;
        }
    }
    if(del == 0) dep[node] = -1;
    return del;
}
int dinic() {
    int ans = 0;
    while(bfs()) {
        memcpy(cur, head, sizeof(head));
        ans += dfs(S, N);
    }
    return ans;
}
void solve(){
    memset(head, -1, sizeof(head));
    lt.clear(); rt.clear(); v.clear();
    tot = 0;
    int n = gn();
    for(int i = 1; i <= n; ++i) {
        ll x = gl(), y = gl();
        v.pb({x - y, x + y});
        lt.pb(x - y);
        rt.pb(x + y);
    }
    sort(lt.begin(), lt.end());
    sort(rt.begin(), rt.end());
    lt.erase(unique(lt.begin(), lt.end()), lt.end());
    rt.erase(unique(rt.begin(), rt.end()), rt.end());
    S = n + n + 1, T = S + 1;
    for(int i = 0, len = lt.size(); i < len; ++i) {
        add(i + 1, S, 0);
        add(S, i + 1, 1);
    }
    for(int i = 0, len = rt.size(); i < len; ++i) {
        add(i + 1 + n, T, 1);
        add(T, i + 1 + n, 0);
    }
    for(int i = 0; i < n; ++i) {
        add(where(v[i].first, 1), where(v[i].second, 0) + n, 1);
        add(where(v[i].second, 0) + n, where(v[i].first, 1), 0);
    }
    print(dinic()), putchar(10);
}
```

### 欧拉回路

对于欧拉回路的判断：
无向图G存在欧拉通路的充要条件是：
$G$为连通图，并且$G$仅有两个奇度结点（度数为奇数的顶点）或者无奇度结点。
推论1：
1.  当$G$是仅有两个奇度结点的连通图时，$G$的欧拉通路必以此两个结点为端点。
2.  当$G$是无奇度结点的连通图时，$G$必有欧拉回路。
3. $G$为欧拉图（存在欧拉回路）的充分必要条件是$G$为无奇度结点的连通图。

有向图$D$存在欧拉通路的充要条件是：
$D$为有向图，$D$的基图连通，并且所有顶点的出度与入度都相等；或者除两个顶点外，其余顶点的出度与入度都相等，而这两个顶点中一个顶点的出度与入度之差为1，另一个顶点的出度与入度之差为-1。

推论2：
1. 当D除出 入度之差为1，-1的两个顶点之外，其余顶点的出度与入度都相等时，D的有向欧拉通路必以出、入度之差为1的顶点作为始点，以出、入度之差为-1的顶点作为终点。
2. 当D的所有顶点的出 入度都相等时，D中存在有向欧拉回路。
3. 有向图D为有向欧拉图的充分必要条件是D的基图为连通图，并且所有顶点的出、入度都相等。

```cpp
//判断欧拉回路
int main(){
    memset(degree, 0, sizeof(degree));
    int n, m;
    cin >> n >> m;
    for(int i = 1; i <= n; ++i) f[i] = i;
    for(int i = 0; i < m; ++i) {
        int x,y;
        cin >> x >> y;
        unit(x, y);
        ++degree[x]; ++degree[y];
    }
    int x = 0,sum = 0;
    for(int i = 1; i <= n; ++i) {
        if(f[i] == i) ++sum;
        if(degree[i] & 1) ++x;
    }
    if(x != 2 && x != 0 || sum != 1) printf("YES");
    else printf("NO");
}

//求解欧拉回路
void eular(int x){
    for(int i = 1;i <= n; ++i) {
        if(mp[x][i]) {
            mp[x][i]--;
            mp[i][x]--;
            dfs(i);
        }
    }
    path.push(i);
}
```

### 拓扑排序

```cpp
//BFS实现
int degree[N];
vector<int> v[N];
bool topo(int n){
    queue<int> Q;
    int cnt=0;
    for(int i = 1; i <= n; ++i) {
        if(!degree[i]) Q.push(i);
    }
    while(!Q.empty()){
        int now = Q.front();
        Q.pop();
        ++cnt;
        for(int i = 0,len = v[now].size(); ++i) {
            int m = v[now][i];
            --degree[m];
            if(!degree[m]) Q.push(m);
        }
    }
    return cnt == n;
}

//DFS实现
bool dfs(int n) {
    vis[n] = -1;
    for(int i = 0, len = v[n].size(); i < len; ++i) {
        int m = v[n][i];
        if(vis[m] == -1) return false;
        if(!vis[m] && !dfs(m)) return false;
    }
    vis[n] = 1;
    ans.push(n);
    return true;
}
```

## 动态规划

### 分组背包

```jsx
struct EulerSieve{
    static const int N = 3e4 + 50;
    int vis[N], prime[N];
    int cnt = 0;

    void getPrime () {
        vis[1] = 1;
        for(int i = 2; i < N; ++i) {
            if(!vis[i]) prime[++cnt] = i;
            for(int j = 1; j <= cnt; ++j) {
                if(i * prime[j] > N) break;
                vis[i * prime[j]] = 1;
                if(i % prime[j] == 0) break;
            }
        }
    }

} phi;

struct Groupbackpack {
    static const int N = 3e4 + 50;
    double dp[N];
    int num;

    void init() {
        phi.getPrime();
        for(int i = 1; i < N; ++i) ln[i] = log(i);
        for(int i = 0; i < N; ++i) dp[i] = 0;
        num = phi.cnt;
        getans();
    }

    // dp[i][j] = dp[i - 1][j - w] + log(k)
    void getans() {
        dp[1] = 0.0;
        for(int i = 1; i <= num; ++i) {
            for(int j = N - 1; j >= 1; --j) {// 先枚举j 再枚举k
                for(int k = phi.prime[i]; k < N; k *= phi.prime[i]) {
                    if(j < k) break;
                    dp[j] = max(dp[j], dp[j - k] + ln[k]);
                }
            }
        }
    }
}dp;
```

### 数位dp

```cpp
//接下来n行 每行一个数字x 接下来一个数len表示数字x在数字串中连续出现的次数不能大于len
const int N = 2e5 + 100;
const int mod = 20020219;
int f[20][15][20][2];
int limit[20];
int a[20];
ll dp(int pos, int num, int cnt, bool flag){
//pos 还剩下几位 num 当前数位上是什么数字 cnt 当前数字出现次数 flag 当前位是否有限制
    if(pos == 0) return cnt <= limit[num];
    if(cnt > limit[num]) return 0;
    if(flag && f[pos][num][cnt][flag] != -1) return f[pos][num][cnt][flag];
    int x = flag ? 9 : a[pos];
    ll ans = 0;
    for(int i = 0; i <= x; ++i) {
        if(i == num)ans = (ans + dp(pos - 1, num, cnt + 1, flag || i < x)) % mod;
        else ans = (ans + dp(pos - 1, i, 1, flag || i < x)) % mod;
    }
    ans = ans % mod;
    if(flag) f[pos][num][cnt][flag] = ans;
    return ans;
}
//计算每一位数的限制
ll calc(ll num){
    memset(f, -1, sizeof(f));
    int pos = 0;
    while(num){
        a[ ++pos ] = num % 10;
        num /= 10;
    }
    return dp(pos, 0, 0, 0);
}

void solve(){
    memset(limit, 0x3f, sizeof(limit));
    ll l = gl(), r = gl(), n = gl();
    for(int i = 1; i <= n; ++i) {
        int x = gn(), y = gn();
        limit[x] = min(limit[x], y);
    }
}
```

### 区间dp

```cpp
//区间dp
void solve(){
    memset(dp, 0x3f, sizeof(dp))；
    for(int i = 1; i <= n; ++i) {
        dp[i][i] = 0;
    }
    for(int l = 2; l <= n; ++l) {
        for(int i = 1; i + l - 1 <= n; ++i) {
            int j = i + l - 1;
            for(int k = i; k < j; ++k) {
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + sum[j]-sum[i - 1]);
            }
        }
    }
}
//优化
int m[2005][2005] = {0};
int sum[1005] = {0};

void solver() {
    memset(dp, 0x7f, sizeof(dp));
    int n = gn();
    for(int i = 1; i <= n; ++i) {
        a[i] = gn();
        sum[i] = sum[i-1] + a[i];
    }
    for(int i = 1; i <= 2 * n; ++i) {
        dp[i][i] = 0;
        m[i][i] = i;
    }
    for(int l = 2;l <= n; ++l) {
        for(int i = 1; i + l - 1 <= 2 * n; ++i) {
            int j = i + l - 1;
            for(int k = m[i][j-1]; k <= m[i + 1][j]; ++k) {
                int a, b;
                if(i - 1 > n) {
                    a = sum[n] + sum[i - 1 - n];
                }
                else a = sum[i - 1];
                if(j > n) {
                    b = sum[n] + sum[j - n];
                } else b = sum[j];
                int t = b - a;
                if(dp[i][j] > dp[i][k] + dp[k + 1][j] + t) {
                    dp[i][j] = dp[i][k] + dp[k + 1][j] + t;
                    m[i][j] = k;
                }
            }
        }
    }
    int ans = 1e9 + 7;
    for(int i = 1; i <= n; ++i) {
        ans=min(ans, dp[i][i + n - 1]);
    }
    cout << ans << endl;
}
```

### 背包问题

```cpp
int n, m;
int N[555], V[555];
int dp[100000 + 1024];
//01背包
void ZeroOnePack(int cost, int weight){
    for(int i = m; i >= cost; --i) {//反向更新
        dp[i] = max(dp[i], dp[i - cost] + weight);
        if(dp[i - cost] + weight == dp[i])//统计方案数
            sum[i] += sum[i-cost];
    }
}
//完全背包
void CompletePack(int cost, int weight){
    for(int i = cost; i <= m; ++i) {
        dp[i] = max(dp[i], dp[i - cost] + weight);
    }
}
//多重背包
void MultiplePack(int cost, int weight, int amount) {
    if(cost * amount > m) {
        CompletePack(cost, weight);
        return ;
    }
    int k = 1;
    while(k < amount) {
        ZeroOnePack(k * cost, k * weight);
        amount -= k;
        k *= 2;
    }
    ZeroOnePack(amount * cost, amount * weight);
}

for(all group k)
    for(v = V....0)
        for(all i belong to group k)
            f[v] = max{f[v], f[v-c[i]]+w[i]}
```

### 最长公共子序列(O(nlogn))

```cpp
int dp[N];
int a[N], b[N];

void solve() {
    int n = gn();
    int k;
    for(int i = 1; i <= n; ++i) k = gn(),a[k] = i;
    for(int i = 1; i <= n; ++i) k = gn(),b[i] = a[k];
    int len = 0;
    fill(dp, dp + N, 0);
    for(int i = 1; i <= n; ++i) {
        if(b[i] > dp[len]){
            dp[++len]=b[i];
        }
        else{
            int l = 1,r = len;
            while(l <= r){
                int mid = (l + r) / 2;
                if(b[i] > dp[mid])l = mid + 1;
                else r = mid - 1;
            }
            dp[l]=min(b[i], dp[l]);
        }
    }
    cout<<len<<endl;
}
```

## 数据结构

### 线段树合并+CDQ分治统计答案

```cpp
#include<bits/stdc++.h>
using namespace std;

#define ll long long

template <typename T>
inline T read() {
    T s = 0,f = 1; char ch = getchar();
    while(!isdigit(ch)) {if(ch == '-') f = -1; ch = getchar();}
    while(isdigit(ch)) {s = (s << 3) + (s << 1) + ch - 48; ch = getchar();}
    return s * f;
}
#define gn() read<int>()
#define gl() read<ll>()

const int N = 1e5 + 100;
const int mod = 1e9 + 7;

vector<int> v[N];

ll ans[N]; int n, q;

struct SegmentTree {
    static const int maxn = 1e5 + 100;
    #define lson(x) s[x].lc
    #define rson(x) s[x].rc
    struct node {
        int lc, rc;
        ll val[2];
    } s[maxn * 80];

    int root[maxn];
    int tot = 0;

    void insert(int &root, int l, int r, int idx, ll val, int sign) {
        if(!root) root = ++tot;

        s[root].val[sign] += val;

        if(l == r) return ;
        int mid = l + r >> 1;

        if(idx <= mid) insert(lson(root), l, mid, idx, val, sign);
        else insert(rson(root), mid + 1, r, idx, val, sign);
    }

    ll query(int root, int l, int r, int L, int R, int sign) {
        if(!root) return 0LL;
        if(L <= l and R >= r) {
            return s[root].val[sign];
        }
        int mid = l + r >> 1;
        ll sum = 0;
        if(L <= mid) sum += query(lson(root), l, mid, L, R, sign);
        if(R > mid) sum += query(rson(root), mid + 1, r, L, R, sign);
        return sum;
    }

    void merge(int &u, int v) {
        if(not u or not v) {
            u += v;
            return ;
        }

        s[u].val[1] += s[v].val[1];
        s[u].val[0] += s[v].val[0];

        merge(lson(u), lson(v));
        merge(rson(u), rson(v));
    }

    void cdq(int node, int L, int R, int l, int r) {
        if(not L or not R) return ;
        ans[node] += s[lson(L)].val[1] * s[rson(R)].val[0];
        if(L != R) ans[node] += s[lson(R)].val[1] * s[rson(L)].val[0];
        if(l == r) return ;
        int mid = l + r >> 1;
        cdq(node, lson(L), lson(R), l, mid);
        cdq(node, rson(L), rson(R), mid + 1, r);
    }
} tree;

void dfs(int node, int fa) {
    tree.cdq(node, tree.root[node], tree.root[node], 1, q);
    for(auto to : v[node]) {
        if(to == fa) continue;
        dfs(to, node);
        ans[node] += ans[to];
        tree.cdq(node, tree.root[node], tree.root[to], 1, q); // cdq分治统计答案
        tree.merge(tree.root[node], tree.root[to]); //区间合并
    }
}

int main() {
    n = gn();
    for(int i = 2; i <= n; ++i) {
        int x = gn();
        v[x].emplace_back(i);
        v[i].emplace_back(x);
    }
    q = gn();
    for(int i = 1; i <= q; ++i) {
        int type = gn(), u = gn(), d = gn();
        if(type == 1) {
            tree.insert(tree.root[u], 1, q, i, d, 1);
        } else {
            tree.insert(tree.root[u], 1, q, i, d, 0);
        }
    }
    dfs(1, 0);
    for(int i = 1; i <= n; ++i) {
        printf("%lld\n", ans[i]);
    }
}
```

### Splay解决区间翻转

```cpp
struct Splay {
    static const int maxn = 1e3 + 100;

    #define lson(x) spl[x].ch[0]
    #define rson(x) spl[x].ch[1]

    struct node {
        int ch[2], siz, fa, rev, minx;
        ll add, sum, val;
    }spl[maxn];

    int tot = 0, st[maxn];

    bool ident(int x, int f) {
        return rson(f) == x;
    }

    void connect(int x, int f, int s) {
        spl[f].ch[s] = x;
        spl[x].fa = f;
    }

    void update(int x) {
        spl[x].siz = 1; spl[x].sum = spl[x].minx = spl[x].val;
        if(lson(x)) {
            spl[x].siz += spl[lson(x)].siz, spl[x].sum += spl[lson(x)].sum;
            spl[x].minx = min(spl[x].minx, spl[lson(x)].minx);
        }
        if(rson(x)) {
            spl[x].siz += spl[rson(x)].siz, spl[x].sum += spl[rson(x)].sum;
            spl[x].minx = min(spl[x].minx, spl[rson(x)].minx);
        }
    }

    void revtag(int x) {
        if(not x) return ;
        swap(lson(x), rson(x));
        spl[x].rev ^= 1;
    }

    void addtag(int x, int val) {
        if(not x) return ;
        spl[x].val += val, spl[x].sum += val * spl[x].siz, spl[x].add += val, spl[x].minx += val;
    }

    void spread(int x) {
        if(spl[x].rev) {
            if(lson(x)) revtag(lson(x));
            if(rson(x)) revtag(rson(x));
            spl[x].rev = 0;
        }
        if(spl[x].add) {
            if(lson(x)) addtag(lson(x), spl[x].add);
            if(rson(x)) addtag(rson(x), spl[x].add);
            spl[x].add = 0;
        }
    }

    void rotate(int x) {
        int f = spl[x].fa, ff = spl[f].fa, k = ident(x, f);
        connect(spl[x].ch[k ^ 1], f, k);
        connect(x, ff, ident(f, ff));
        connect(f, x, k ^ 1);
        update(f), update(x);
    }

    void splaying(int x, int to) {
        int y = x, top = 0; st[++top] = y;
        while(spl[y].fa) st[++top] = spl[y].fa, y = spl[y].fa;
        while(top) spread(st[top--]); // ***
        while(spl[x].fa != to) {
            int f = spl[x].fa, ff = spl[f].fa;
            if(ff == to) rotate(x);
            else if(ident(x, f) == ident(f, ff)) rotate(f), rotate(x);
            else rotate(x), rotate(x);
        }
        update(x);// ***
    }

    int build(int fa, int l, int r) {
        if(l > r) return 0;
        int mid = l + r >> 1;
        int now = ++tot;
        spl[now].val = data[mid], spl[now].fa = fa, spl[now].rev = spl[now].add = 0;
        lson(now) = build(now, l, mid - 1);
        rson(now) = build(now, mid + 1, r);
        update(now);
        return now;
    }

    int kth(int x) {
        int now = spl[0].ch[1];
        while(true) {
            spread(now); // *
            int sum = spl[lson(now)].siz + 1;
            if(x == sum) return now;
            if(sum > x) now = lson(now);
            else {
                x -= sum;
                now = rson(now);
            }
        }
    }
    
    void insert(int pos, int x) {
        int l = kth(pos + 1), r = kth(pos + 2);
        splaying(l, 0), splaying(r, l);
        spl[++tot].val = x, spl[x].fa = r, lson(r) = tot;
        update(tot); update(r), update(l);
    }
    
    void Delete(int pos) {
        int l = kth(pos), r = kth(pos + 2);
        splaying(l, 0), splaying(r, l);
        lson(r) = 0; update(r), update(l);
    }

    void turn(int l, int r) {
        l = kth(l), r = kth(r + 2);
        splaying(l, 0);
        splaying(r, l);
        revtag(lson(r));
    }

    void add(int l, int r, ll val) {
        l = kth(l), r = kth(r + 2);
        splaying(l, 0);
        splaying(r, l);
        addtag(lson(r), val);
    }
    
    ll getmin(int l, int r) {
        l = kth(l), r = kth(r + 2);
        splaying(l, 0), splaying(r, l);
        return spl[lson(r)].minx;
    }

    ll query(int l, int r) {
        l = kth(l), r = kth(r + 2);
        splaying(l, 0);
        splaying(r, l);
        return spl[lson(r)].sum;
    }

    void print(int node) {
        spread(node);
        if(lson(node)) print(lson(node));
        if(node and spl[node].val != INF and spl[node].val != - INF) cout << spl[node].val << " ";
        if(rson(node)) print(rson(node));
    }

}tree;

int main() {
    int n = gn(), m = gn(), q = gn();
    data[1] = -INF, data[m + 2] = INF;// as soldiers
    for(int j = 1; j <= m; ++j) {
        data[j + 1] = gl();
    }
    tree.spl[0].ch[1] = tree[i].build(0, 1, m + 2);
}
```

### Splay 备用

```cpp
struct SplayTree {
	int fa[MAXN], ch[MAXN][2], val[MAXN], addv[MAXN], siz[MAXN], rev[MAXN], mn[MAXN], sum[MAXN];
	int st[MAXN], root, tot;
	void Rev(int x) {
		if(!x) return;
		swap(ch[x][0], ch[x][1]);
		rev[x] ^= 1;
	}
	void Add(int x, int C) {
		if(!x) return;
		val[x] += C; mn[x] += C; addv[x] += C; 
		sum[x] += C * siz[x];
	}
	void PushDown(int x) {
		if(rev[x]) {
			if(ch[x][0]) Rev(ch[x][0]);
			if(ch[x][1]) Rev(ch[x][1]);
			rev[x] ^= 1;
		}
		if(addv[x]) {
			if(ch[x][0]) Add(ch[x][0], addv[x]);
			if(ch[x][1]) Add(ch[x][1], addv[x]);
			addv[x] = 0;
		}
	}
	void PushUp(int x) {
		siz[x] = 1; sum[x] = mn[x] = val[x]; 
		if(ch[x][0]) siz[x] += siz[ch[x][0]], mn[x] = min(mn[x], mn[ch[x][0]]), sum[x] += sum[ch[x][0]];
		if(ch[x][1]) siz[x] += siz[ch[x][1]], mn[x] = min(mn[x], mn[ch[x][1]]), sum[x] += sum[ch[x][1]];
	}
	void rotate(int x) {
		int y = fa[x], z = fa[y], k = ch[y][1] == x, w = ch[x][!k];
		if(fa[y]) ch[z][ch[z][1]==y] = x; 
		ch[x][!k] = y; ch[y][k] = w;
		if(w) fa[w] = y;
		fa[x] = z; fa[y] = x; 
		PushUp(y); PushUp(x);
	}
	void Splay(int x, int goal) {
		int y = x, top = 0; st[++top] = y;
		while(fa[y]) st[++top] = fa[y], y = fa[y];
		while(top) PushDown(st[top--]);
		while(fa[x] != goal) {
			int y = fa[x], z = fa[y];
			if(fa[y] != goal) rotate((ch[z][1]==y)^(ch[y][1]==x) ? x : y);
			rotate(x);
		}
		if(!goal) root = x;
		PushUp(x);
	}
	int kth(int k) {
		int x = root, cur;
		while(true) {
			PushDown(x);
			cur = siz[ch[x][0]] + 1;
			if(cur == k) return x;
			if(k < cur) x = ch[x][0];
			else k -= cur, x = ch[x][1];
		}
	}
	int Build(int l, int r, int pre, int *a) {
		int x = ++tot, mid = (l + r) >> 1;
		fa[x] = pre; val[x] = a[mid];
		if(l < mid) ch[x][0] = Build(l, mid-1, x, a);
		if(r > mid) ch[x][1] = Build(mid+1, r, x, a);
		PushUp(x);
		return x;
	}
	void Reverse(int x, int y) {
		x = kth(x); y = kth(y+2);
		Splay(x, 0); Splay(y, x); Rev(ch[y][0]);
	}
	void Insert(int pos, int x) {
		int pos1 = kth(pos+1), pos2 = kth(pos+2);
		Splay(pos1, 0); Splay(pos2, pos1);
		val[++tot] = x; fa[tot] = pos2; ch[pos2][0] = tot;
		PushUp(tot); PushUp(pos2); PushUp(pos1);
	}
	void Delete(int pos) {
		int x = kth(pos), y = kth(pos+2);
		Splay(x, 0); Splay(y, x);
		ch[y][0] = 0; PushUp(y); PushUp(x);
	}
	void Add(int x, int y, int C) {
		x = kth(x); y = kth(y+2);
		Splay(x, 0); Splay(y, x); Add(ch[y][0], C);
	}
	int GetMin(int x, int y) {
		x = kth(x); y = kth(y+2);
		Splay(x, 0); Splay(y, x);
		return mn[ch[y][0]];
	}
	int GetSum(int x, int y) {
		x = kth(x); y = kth(y + 2);
		Splay(x, 0); Splay(y, x);
		return sum[ch[y][0]];
	}
	void OutPut(int x, vector<int> &vec) {
		PushDown(x);
		if(ch[x][0]) OutPut(ch[x][0], vec);
		vec.push_back(val[x]);
		if(ch[x][1]) OutPut(ch[x][1], vec);
	}
	void Build(int n, int *a) {
		root = Build(0, n+1, 0, a);
	}
}seq[MAXN];
```

### 线段树合并（可解决树上问题 灵活运用）

```cpp
struct SegmentTree {
    static const int maxn = 1e5 + 100;
	#define lson(x) s[x].lc
	#define rson(x) s[x].rc
    struct node {
        int lc, rc, sum;
    }s[maxn * 80];

    int tot = 0, root[maxn];

    void insert(int &now, int l, int r, int idx, int val) {
        if(!now) now = ++tot;
        s[now].sum += val;
        if(l == r) return ;
        int mid = l + r >> 1;
        if(idx <= mid) insert(lson(now), l, mid, idx, val);
        else insert(rson(now), mid + 1, r, idx, val);
    }

    int query(int now, int l, int r, int L, int R) {
        if(!now) return 0;
        if(L <= l and R >= r) return s[now].sum;
        int mid = l + r >> 1;
        int sum = 0;
        if(L <= mid) sum += query(lson(now), l, mid, L, R);
        if(R > mid) sum += query(rson(now), mid + 1, r, L, R);
        return sum;
    }

    int merge(int u, int v) {
        if(not u or not v) return u + v;
        int t = ++tot;
        s[t].sum = s[u].sum + s[v].sum;
        s[t].lc = merge(lson(u), lson(v));
        s[t].rc = merge(rson(u), rson(v));
        return t;
    }

    void merge_ (int &u, int v) {
        if(not u or not v) {
            u += v;
            return ;
        }
        s[u].sum += s[v].sum;
        merge_(lson(u), lson(v));
        merge_(rson(u), rson(v));
    }
} tree;

void dfs(int node, int fa) {
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(to == fa) continue;
        dfs(to, node);
        tree.merge_(tree.root[node], tree.root[to]);
    }
    int l = where(p[node]) + 1;
    if(l <= len) ans[node] = tree.query(tree.root[node], 1, len, l, len);
    else ans[node] = 0;
}
```

### 基础splay（名次树）

```cpp
struct Splay {
    static const int maxn = 1e5+5;

    struct Node {
        int fa, ch[2], val, cnt, size;
    }spl[maxn];

    int tot = 0;
    // update size
    inline void update(int x) {
        spl[x].size = spl[x].cnt + spl[spl[x].ch[0]].size + spl[spl[x].ch[1]].size; 
    }
    // judge fa and son
    inline bool ident(int x,int f) {
        return spl[f].ch[1] == x;
    }
    // build fa and son
    inline void connect(int x,int f,int s) {
        spl[f].ch[s] = x;
        spl[x].fa = f;
    }
    // rotate and update
    void rotate(int x) {
        int f=spl[x].fa, ff=spl[f].fa, k=ident(x, f);
        connect(spl[x].ch[k ^ 1], f, k);
        connect(x, ff, ident(f,ff));
        connect(f, x, k ^ 1);
        update(f), update(x);
    }
    // rotate to root
    void splaying(int x, int to) {
        while(spl[x].fa != to) {
            int f = spl[x].fa,ff = spl[f].fa;
            if(ff == to) rotate(x);
            else if(ident(x, f) == ident(f, ff)) {
                rotate(f), rotate(x);
            } else rotate(x), rotate(x);
        }
    }
    // build node
    void newnode(int &now, int fa, int val) {
        now = ++tot;
        spl[now].val = val;
        spl[now].fa = fa;
        spl[now].size = spl[now].cnt = 1;
    }
    // insert node
    void insert(int x) {
        int now = spl[0].ch[1];
        if(!now) {
            newnode(spl[0].ch[1], 0, x);
        } else {
            while (true) {
                ++spl[now].size;
                if(spl[now].val == x) {
                    ++spl[now].cnt;
                    splaying(now, 0);
                    return ;
                }
                int nxt = (x < spl[now].val ? 0 : 1);
                if(!spl[now].ch[nxt]) {
                    newnode(spl[now].ch[nxt], now, x);
                    splaying(spl[now].ch[nxt], 0);
                    return ;
                }
                now = spl[now].ch[nxt];
            }
        }
    }
    // del node
    void del(int x) {
        int pos = find(x);
        if(!pos) return ;
        if(spl[pos].cnt > 1) {
            spl[pos].cnt--;
            spl[pos].size--;
        } else {
            if(!spl[pos].ch[0] and !spl[pos].ch[1]) spl[0].ch[1] = 0;
            else if(!spl[pos].ch[0]) {
                spl[0].ch[1] = spl[pos].ch[1];
                spl[spl[pos].ch[1]].fa = 0;
            } else {
                int left = spl[pos].ch[0];
                while(spl[left].ch[1]) left = spl[left].ch[1];
                splaying(left, 0);

                connect(spl[pos].ch[1], left, 1);
                update(left);
            }
        }
    }
    // find pos
    int find(int x) {
        int now = spl[0].ch[1];
        while(true) {
            if(x == spl[now].val) {
                splaying(now, 0);
                return now;
            }
            now = spl[now].ch[x > spl[now].val];
            if(!now) return 0;
        }
    }
    // find this num x's rank
    int rank(int x) {
        int pos = find(x);
        return spl[spl[pos].ch[0]].size + 1;
    }
    // find who is rank x
    int arank(int x) {
        int now = spl[0].ch[1];
        while (true) {
            int num = spl[now].size - spl[spl[now].ch[1]].size;
            if(x > spl[spl[now].ch[0]].size and x <= num) {
                splaying(now, 0);
                return spl[now].val;
            }
            if(x < num) now = spl[now].ch[0];
            else x -= num, now = spl[now].ch[1];
        }

    }
    // find pre now
    int pre(int x) {
        insert(x);
        int now = spl[spl[0].ch[1]].ch[0];
        if(!now) return -1;
        while(spl[now].ch[1]) now = spl[now].ch[1];
        del(x);
        return now;
    }
    // find nxt now
    int nxt(int x) {
        insert(x);
        int now = spl[spl[0].ch[1]].ch[1];
        if(!now) return -1;
        while(spl[now].ch[0]) now = spl[now].ch[0];
        del(x);
        return now;
    }
} splay;
```

### 带撤销并查集+离线

```cpp
struct node {
    int l, r, w;
} s[N];

int tot = 0, ans[N];

struct star {
    int id, x, y, w;
} qt[N];

int cnt = 0;

struct DSU {
    static const int maxn = 5e5 + 100;
    int f[maxn], dep[maxn];
    int siz = 0;
    struct node {
        int son, fa, prefa, dep;
    } st[maxn];

    void init() {
        for(int i = 1; i < maxn; ++i) {
            f[i] = i;
            dep[i] = 1;
        }
    }

    int found(int x) {
        return f[x] == x ?  x : found(f[x]);
    }

    int unit(int x, int y, int sign) {
        x = found(x), y = found(y);
        if(x == y) return 1;
        if(dep[x] > dep[y]) swap(x, y);
        int fax = found(x), fay = found(y);
        if(sign == 1) st[++siz] = {fax, fay, fax, dep[y]};
        if(dep[y] == dep[x]) dep[y]++;
        f[fax] = fay;
        return 0;
    }

    void del() {
        for(int i = siz; i >= 1; --i) {
            int son = st[i].son, fa = st[i].fa;
            f[son] = st[i].prefa;
            dep[fa] = st[i].dep;
        }
        siz = 0;
    }
} dsu;

int main() {
    int n = gn(), m = gn();

    for(int i = 1; i <= m; ++i) {
        s[i] = {gn(), gn(), gn()};
    }

    int q = gn();
    for(int i = 1; i <= q; ++i) {
        int k = gn();
        for(int j = 1; j <= k; ++j) {
            int id = gn();
            qt[++cnt] = {i, s[id].l, s[id].r, s[id].w};
        }
    }
    sort(s + 1, s + 1 + m, [](node a, node b) {
        return a.w < b.w;
    });
    sort(qt + 1, qt + 1 + cnt, [](star a, star b) {
        if(a.w == b.w) return a.id < b.id;
        return a.w < b.w;
    });
    dsu.init();
    int now = 1;
    for(int i = 1; i <= m; ++i) {
        while(qt[now].w == s[i].w && now <= cnt) {
            if(ans[qt[now].id] == 1) {
                ++now; continue;
            }
            int flag = dsu.unit(qt[now].x, qt[now].y, 1), id = qt[now].id;
            while(qt[now + 1].id == id && qt[now + 1].w == s[i].w) {
                ++now;
                flag += dsu.unit(qt[now].x, qt[now].y, 1);
            }
            if(flag) ans[id] = 1;
            dsu.del();
            ++now;
        }
        dsu.unit(s[i].l, s[i].r, 0);
    }
    for(int i = 1; i <= q; ++i) {
        ans[i] == 1 ? printf("NO\n") : printf("YES\n");
    }
}
```

### 线段树离线维护$Mex$ （在线主席树 思路相同）

```cpp
constexpr int N = 1e5 + 100;

int a[N], pre[N], ans[N];

struct SegmentTree {
    static const int maxn = 1e5 + 100;
    #define lson node << 1
    #define rson node << 1 | 1
    int minx[maxn << 2];

    void pushup(int node, int l, int r) {
        minx[node] = min(minx[lson], minx[rson]);
    }

    void insert(int node, int l, int r, int idx, int val) {
        if(l == r) {
            minx[node] = val;
            return ;
        }
        int mid = l + r >> 1;
        if(idx <= mid) insert(lson, l, mid, idx, val);
        else insert(rson, mid + 1, r, idx, val);
        pushup(node, l, r);
    }

    int query(int node, int l, int r, int L, int R) {
        if(L == l && R == r) {
            return minx[node];
        }
        int mid = l + r >> 1;
        if(R <= mid) return query(lson, l, mid, L, R);
        else if(L > mid) return query(rson, mid + 1, r, L, R);
        else return min(query(lson, l, mid, L, mid), query(rson, mid + 1, r, mid + 1, R));
    }
} tree;

int main() {
    for(int i = 1; i <= n; ++i) {
        if(a[i] != 1) {
            if(tree.query(1, 1, n, 1, a[i] - 1) > pre[a[i]]) {
                ans[a[i]] = 1;
            }
        }
        tree.insert(1, 1, n, a[i], i);
        pre[a[i]] = i;
        if(a[i] > 1) ans[1] = 1;
    }

    for(int i = 2; i <= n + 1; ++i) {
        if(tree.query(1, 1, n, 1, i - 1) > pre[i]) {
            ans[i] = 1;
        }
    }
}
```

### 线段树维护带权中位数

```cpp
const int N = 2e5 + 100;
const int mod = 1e9 + 7;

ll w[N];

struct Segment {
	static const int MAX = 2e5 + 100;
	#define lson node << 1
	#define rson node << 1 | 1
	ll sum[MAX << 2];

	void pushup(int node, int l, int r) {
		sum[node] = sum[lson] + sum[rson];
	}

	void build(int node, int l, int r) {
		if(l == r) {
			sum[node] = w[l];
			return ;
		}
		int mid = l + r >> 1;
		build(lson, l, mid);
		build(rson, mid + 1, r);
		pushup(node, l, r);
	}

	void change(int node, int l, int r, int idx, int val) {
		if(l == r) {
			sum[node] = val;
			return ;
		}
		int mid = l + r >> 1;
		if(idx <= mid) change(lson, l, mid, idx, val);
		else change(rson, mid + 1, r, idx, val);
		pushup(node, l, r);
	}

	int queryid(int node, int l, int r, int L, int R, ll val) {
		if(l == r) return l;
		int mid = l + r >> 1;
		if(R <= mid) return queryid(lson, l, mid, L, R, val);
		else if(L > mid) return queryid(rson, mid + 1, r, L, R, val);
		else {
            ll lsum = querysum(lson, l, mid, L, mid);
			if(lsum >= val) return queryid(lson, l, mid, L, mid, val);
			else return queryid(rson, mid + 1, r, mid + 1, R, val - lsum);
		}
	}

	ll querysum(int node, int l, int r, int L, int R) {
		if(L <= l && R >= r) {
			return sum[node];
		}
		int mid = l + r >> 1;
		ll val = 0;
		if(L <= mid) val += querysum(lson, l, mid, L, R);
		if(R > mid) val += querysum(rson, mid + 1, r, L, R);
		return val;
	}
} treeans;

int main() {
	treeans.build(1, 1, n);
	for(int i = 1; i <= m; ++i) {
		ll l = gl(), r = gl();
        ll k = treeans.querysum(1, 1, n, l, r);
        k = (k + 1) / 2;
        ll id = treeans.queryid(1, 1, n, l, r, k);
	}
}
```

### 扩展域并查集

```cpp
// from 1 to n express good, from n + 1 to n + n express bad
using namespace std;
const int N = 1e5+500;
const int M = 2e4+500;

struct node{
    int x, y;
    long long w;
    bool operator < (const node &rhs)const {
        return w > rhs.w;
    }
}s[N];

int f[M*2];

void init(int n) {
    for(int i = 1; i <= n; ++i){
        f[i] = i;
        f[n+i] = n+i;
    }
}

int found(int x) {
    if(f[x] == x) return x;
    return f[x] = found(f[x]);
}

bool isunit(int x, int y) {
    x = found(x);
    y = found(y);
    if(x == y) return true;
    return false;
}

void unit(int x,int y) {
    x = found(x);
    y = found(y);
    f[x]=y;
}

int n,m;

int main(){
    for(int i = 1;i <= m; ++i) {
        if(s[i].x==s[i].y)continue;
        if(!isunit(s[i].x, s[i].y)) {
            unit(s[i].x, s[i].y+n);
            unit(s[i].x+n, s[i].y);
        }
    }
}
```



### 二维树状数组（区间修改 + 单点查询）

二维前缀和：$sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + a[i][j]$

我们可以令差分数组 $d[i][j]$ 表示$a[i][j]$ 与 $a[i - 1][j] + a[i][j - 1] - a[i - 1][j - 1]$的差。

```cpp
struct TreeArray {
    static const int maxn = 1e3 + 100;
    int tree[maxn][maxn];
    int n;

    int lowbit(int x) {
        return -x & x;
    }

    void add(int x, int y, int val) {
        while(x <= n) {
            int ty = y;
            while(ty <= n) {
                tree[x][ty] += val;
                ty += lowbit(ty);
            }
            x += lowbit(x);
        }
    }
    
    void intervaladd(int x1, int y1, int x2, int y2, int val) {
        add(x1, y1, val);
        add(x1, y2 + 1, -val);
        add(x2 + 1, y1, -val);
        add(x2 + 1, y2 + 1, val);
    }

    int ask(int x, int y) {
        int res = 0;
        while(x) {
            int ty = y;
            while(ty) {
                res += tree[x][ty];
                ty -= lowbit(ty);
            }
            x -= lowbit(x);
        }
        return res;
    }
};
```



### 二维树状数组（单点修改 + 区间查询）

```cpp
struct TreeArray {
    static const int maxn = 1e3 + 100;
    int tree[maxn][maxn];
    int n;

    int lowbit(int x) {
        return -x & x;
    }

    void add(int x, int y, int val) {
        while(x <= n) {
            int ty = y;
            while(ty <= n) {
                tree[x][ty] += val;
                ty += lowbit(ty);
            }
            x += lowbit(x);
        }
    }

    int ask(int x, int y) {
        int res = 0;
        while(x) {
            int ty = y;
            while(ty) {
                res += tree[x][ty];
                ty -= lowbit(ty);
            }
            x -= lowbit(x);
        }
        return res;
    }
};
```



### 线段树+字符串hash结合

```cpp
struct node {
    static const int maxn = 2e5 + 100;
    static const ll base = 131;
    static const ll mod = 1e9 + 7;
    #define lson (node << 1)
    #define rson (node << 1 | 1)
    ll sum[maxn << 2], p[maxn], pp[maxn];
    int lazy[maxn << 2];

    void getp() {
        p[0] = 1; pp[0] = 1;
        for(int i = 1; i < maxn; ++i) {
            p[i] = p[i - 1] * base % mod;
            pp[i] = (pp[i - 1] + p[i]) % mod;
        }
    }

    void pushup(int node, int l, int r) {
        int mid = l + r >> 1;
        sum[node] = (sum[lson] * p[r - mid] % mod + sum[rson]) % mod;
    }

    void spread(int node, int l, int r) {
        if(lazy[node] != -1) {
            int mid = l + r >> 1;
            lazy[lson] = lazy[rson] = lazy[node];
            sum[lson] = (lazy[node] * (pp[mid - l] % mod)) % mod;
            sum[rson] = (lazy[node] * (pp[r - mid - 1] % mod)) % mod;
            lazy[node] = -1;
        }
    }

    void build(int node, int l, int r) {
        lazy[node] = -1;
        if(l == r) {
            sum[node] = a[l];
            return ;
        }
        int mid = l + r >> 1;
        build(lson, l, mid);
        build(rson, mid + 1, r);
        pushup(node, l, r);
    }

    void change(int node, int l, int r, int L, int R, int val) {
        if(L <= l && R >= r) {
            sum[node] = val * pp[r - l] % mod;
            lazy[node] = val;
            return ;
        }
        spread(node, l, r);
        int mid = l + r >> 1;
        if(L <= mid) change(lson, l, mid, L, R, val);
        if(R > mid) change(rson, mid + 1, r, L, R, val);
        pushup(node, l, r);
    }

    ll query(int node, int l, int r, int L, int R) {
        if(L == l && R == r) {
            return sum[node] % mod;
        }
        spread(node, l, r);
        int mid = l + r >> 1;
        if(R <= mid) return query(lson, l, mid, L, R);
        if(L > mid) return query(rson, mid + 1, r, L, R);
        else {
            ll lc = query(lson, l, mid, L, mid);
            ll rc = query(rson, mid + 1, r, mid + 1, R);
            return (lc * p[R - mid] % mod + rc) % mod;
        }
    }
} tree;

int main() {
    int n = gn(), m = gn(), k = gn();
    for(int i = 1; i <= n; ++i) {
        scanf("%1d", &a[i]);
    }
    tree.getp();
    tree.build(1, 1, n);
    int sum = m + k;
    for(int i = 1; i <= sum; ++i) {
        int cmd = gn(), l = gn(), r = gn(), val = gn();
        if(cmd == 1) {
            tree.change(1, 1, n, l, r, val);
        } else {
            if (r - l + 1 <= val) puts("YES");
            else {
                ll sumleft = tree.query(1, 1, n, l, r - val);
                ll sumright = tree.query(1, 1, n, l + val, r);
                if(sumleft == sumright) {
                    puts("YES");
                } else puts("NO");
            }

        }
    }
}
```



### 二维线段树（区间修改 + 单点查询）

```jsx
struct SegmentTree {
    static const int M = 1050;
	#define lson node << 1
	#define rson node << 1 | 1
    // in tree begin
    struct InSegmentTree {
        int maxn[M << 2], minx[M << 2];

        void pushup(int node, int l, int r) {
            maxn[node] = max(maxn[lson], maxn[rson]);
            minx[node] = min(minx[lson], minx[rson]);
        }

        void build(int node, int l, int r, int idx) {
            if(l == r) {
                maxn[node] = minx[node] = mp[idx][l];
                return ;
            }
            int mid = l + r >> 1;
            build(lson, l, mid, idx);
            build(rson, mid + 1, r, idx);
            pushup(node, l, r);
        }

        void singlechange(int node, int l, int r, int idx, int val) {
            if(l == r) {
                maxn[node] = minx[node] = val;
                return ;
            }
            int mid = l + r >> 1;
            if(idx <= mid) singlechange(lson, l, mid, idx, val);
            else singlechange(rson, mid + 1, r, idx, val);
            pushup(node, l, r);
        }

        pair<int, int> intervalquery(int node, int l, int r, int L, int R) {
            if(L == l && R == r) {
                return {maxn[node], minx[node]};
            }
            int mid = l + r >> 1;
            if(R <= mid) return intervalquery(lson, l, mid, L, R);
            else if(L > mid) return intervalquery(rson, mid + 1, r, L, R);
            else {
                pair<int, int> lf, rt;
                lf = intervalquery(lson, l, mid, L, mid);
                rt = intervalquery(rson, mid + 1, r, mid + 1, R);
                return {max(lf.first, rt.first), min(lf.second, rt.second)};
            }
        }

    };

    InSegmentTree tree[M << 2];

    void pushup(int node, int l, int r, InSegmentTree &now, InSegmentTree &lf, InSegmentTree &rt) {
        now.maxn[node] = max(lf.maxn[node], rt.maxn[node]);
        now.minx[node] = min(lf.minx[node], rt.minx[node]);
        if(l == r) return;
        int mid = l + r >> 1;
        pushup(lson, l, mid, now, lf, rt);
        pushup(rson, mid + 1, r, now, lf, rt);
    }

    void build(int node, int l, int r, int idy) {
        if(l == r) {
            tree[node].build(1, 1, idy, l);
            return ;
        }
        int mid = l + r >> 1;
        build(lson, l, mid, idy);
        build(rson, mid + 1, r, idy);
        pushup(1, 1, idy, tree[node], tree[lson], tree[rson]);
    }

    void update(int node, int l, int r, int idx, InSegmentTree &now, InSegmentTree &lf, InSegmentTree &rt) {
        now.maxn[node] = max(lf.maxn[node], rt.maxn[node]);
        now.minx[node] = min(lf.minx[node], rt.minx[node]);
        if(l == r) return ;
        int mid = l + r >> 1;
        if(idx <= mid) update(lson, l, mid, idx, now, lf, rt);
        else update(rson, mid + 1, r, idx, now, lf, rt);
    }
    void singlechange(int node, int l, int r, int idy, int X, int Y, int val) {
        if(l == r) {
            tree[node].singlechange(1, 1, idy, Y, val);
            return ;
        }
        int mid = l + r >> 1;
        if(X <= mid) singlechange(lson, l, mid, idy, X, Y, val);
        else singlechange(rson, mid + 1, r, idy, X, Y, val);
        update(1, 1, idy, Y, tree[node], tree[lson], tree[rson]);
    }

    pair<int, int> intervalquery(int node, int l, int r, int idy, int xL, int xR, int yL, int yR) {
        if(xL == l && xR == r) {
            return tree[node].intervalquery(1, 1, idy, yL, yR);
        }
        int mid = l + r >> 1;
        if(xR <= mid) return intervalquery(lson, l, mid, idy, xL, xR, yL, yR);
        else if(xL > mid) return intervalquery(rson, mid + 1, r, idy, xL, xR, yL, yR);
        else {
            pair<int, int> lf, rt;
            lf = intervalquery(lson, l, mid, idy, xL, mid, yL, yR);
            rt = intervalquery(rson, mid + 1, r, idy, mid + 1, xR, yL, yR);
            return {max(lf.first, rt.first), min(lf.second, rt.second)};
        }
    }
} tr;
```

### 二维线段树(查询)

```jsx
struct TwoSegment {
	#define lson node << 1
    #define rson node << 1 | 1
    // in tree begin
    struct Segment {
        static const int MAX = 350;
        int minx[MAX << 2];

        void pushup(int node, int l, int r) {
            minx[node] = min(minx[lson], minx[rson]);
        }

        void build(int node, int l, int r, int idx) { 
            if(l == r) {
                minx[node] = a[idx][l];
                return ;
            }
            int mid = l + r >> 1;
            build(lson, l, mid, idx);
            build(rson, mid + 1, r, idx);
            pushup(node, l, r);
        }

        int query(int node, int l, int r, int L, int R) {
            if(L == l && R == r) {
                return minx[node];
            }
            int mid = l + r >> 1;
            if(R <= mid) return query(lson, l, mid, L, R);
            if(L > mid) return query(rson, mid + 1, r, L, R);
            else {
                int val = 1e9 + 7;
                val = min(val, query(lson, l, mid, L, mid));
                val = min(val, query(rson, mid + 1, r, mid + 1, R));
                return val;
            }
        }
    };
    // in tree end

    static const int M = 350;
    Segment tree[M << 2];

    void init() {
        memset(tree, 0, sizeof(tree));
    }

    void pushup(int node, int l, int r, Segment &rt, Segment &lc, Segment &rc) {
        if(l == r) {
            rt.minx[node] = min(lc.minx[node], rc.minx[node]);
            return ;
        }
        int mid = l + r >> 1;
        pushup(lson, l, mid, rt, lc, rc);
        pushup(rson, mid + 1, r, rt, lc, rc);
        rt.minx[node] = min(rt.minx[lson], rt.minx[rson]);
    }

    void build(int node, int l, int r, int n) {
        if(l == r) {
            tree[node].build(1, 1, n, l);
            return ;
        }
        int mid = l + r >> 1;
        build(lson, l, mid, n);
        build(rson, mid + 1, r, n);
        pushup(1, 1, n, tree[node], tree[lson], tree[rson]);
    }

    int query(int node, int l, int r, int xL, int xR, int yL, int yR, int n) {
        if(xL <= l && xR >= r) {
            return tree[node].query(1, 1, n, yL, yR);
        }
        int mid = l + r >> 1;
        int ans = 1e9 + 7;
        if(xL <= mid) ans = min(query(lson, l, mid, xL, xR, yL, yR, n), ans);
        if(xR > mid) ans = min(query(rson, mid + 1, r, xL, xR, yL, yR, n), ans);
        return ans;
    }
} tr;
```

### 区间修改

```cpp
struct Segment {
    static const int N = 2e5 + 100;
    #define lson node << 1
    #define rson node << 1 | 1

    struct node {
        int minx, lazy, sum;
    } s[N * 4];

    void spread(int node, int l, int r) {
        if(s[node].lazy) {
            int mid = l + r >> 1;
            s[lson].lazy += s[node].lazy;
            s[rson].lazy += s[node].lazy;
            s[lson].minx += s[node].lazy;
            s[rson].minx += s[node].lazy;
            s[lson].sum += s[node].lazy * (mid - l + 1);
            s[rson].sum += s[node].lazy * (r - mid);
            s[node].lazy = 0;
        }
    }

    void pushup(int node, int l, int r) {
        s[node].sum = s[lson].sum + s[rson].sum;
        s[node].minx = min(s[lson].minx, s[rson].minx);
    }

    void build(int node, int l, int r, int *ar) {
        s[node].lazy = 0;
        if(l == r) {
            s[node].sum = ar[l];
            s[node].minx = ar[l];
            return ;
        }
        int mid = l + r >> 1;
        build(lson, l, mid, ar);
        build(rson, mid + 1, r, ar);
        pushup(node, l, r);
    }
    
    void change(int node, int l, int r, int L, int R, int val) {
        if(L <= l && R >= r) {
            s[node].sum += val * (r - l + 1);
            s[node].minx += val;
            s[node].lazy += val;
            return ;
        }
        spread(node, l, r);
        int mid = l + r >> 1;
        if(L <= mid) change(lson, l, mid, L, R, val);
        if(R > mid) change(rson, mid + 1, r, L, R, val);
        pushup(node, l, r);
    }

    int querysum(int node, int l, int r, int L, int R) {
        if(L <= l && R >= r) {
            return s[node].sum;
        }
        spread(node, l, r);
        int mid = l + r >> 1;
        int val = 0;
        if(L <= mid) val += querysum(lson, l, mid, L, R);
        if(R > mid) val += querysum(rson, mid + 1, r, L, R);
        return val;
    }

    int queryminx(int node, int l, int r, int L, int R) {
        if(L == l && R == r) {
            return s[node].minx;
        }
        spread(node, l, r);
        int mid = l + r >> 1;
        
        if(R <= mid) return queryminx(lson, l, mid, L, R);
        else if(L > mid) return queryminx(rson, mid + 1, r, L, R);
        else {
            int val = 1e9 + 10;
            val = min(val, queryminx(lson, l, mid, L, mid));
            val = min(val, queryminx(rson, mid + 1, r, mid + 1, R));
            return val;
        }
    }
    int queryMinxIndex(int node, int l, int r, int L, int R) {
        if(l == r) {
            return l;
        }
        spread(node, l, r);
        int mid = l + r >> 1;
        if(R <= mid) return queryMinxIndex(lson, l, mid, L, R);
        else if(L > mid) return queryMinxIndex(rson, mid + 1, r, L, R);
        else {
            int val1, val2;
            val1 = queryminx(lson, l, mid, L, mid);
            val2 = queryminx(rson, mid + 1, r, mid + 1, R);
            if (val1 < val2) {
                return queryMinxIndex(lson, l, mid, L, mid);
            } else return queryMinxIndex(rson, mid + 1, r, mid + 1, R);
        }
    }
}s1, s2;
```

### 主席树统计区间里不同的个数

```cpp
const int N = 3e4 + 100;
const int mod = 1e9 + 7;
const int M = 1500;

int n, a[N], root[N], tot = 0, last[1000500], ret = 0;

struct node {
    int lc, rc, sum;
}s[N * 40];

void insert(int l, int r, int pre, int &now, int idx, int c) {
    s[++tot] = s[pre];
    now = tot;
    s[now].sum += c;
    if(l == r) return ;
    int mid = l + r >> 1;
    if(idx <= mid) insert(l, mid, s[pre].lc, s[now].lc, idx, c);
    else insert(mid + 1, r, s[pre].rc, s[now].rc, idx, c);
}

void query(int rt, int l, int r, int p) {
    if(l == r) {
        ret += s[rt].sum;
        return ;
    }
    int mid = l + r >> 1;
    if(p <= mid) ret += s[s[rt].rc].sum, query(s[rt].lc, l, mid, p);
    else query(s[rt].rc, mid + 1, r, p);
}

int main() {
    n = gn();
    for(int i = 1; i <= n; ++i)  a[i] = gn();
    for(int i = 1; i <= n; ++i) {
        if(!last[a[i]]) insert(1, n, root[i - 1], root[i], i, 1);
        else {
            int temp = 0;
            insert(1, n, root[i - 1], temp, last[a[i]], -1);
            insert(1, n, temp, root[i], i, 1);
        }
        last[a[i]] = i;
    }
    int q = gn();
    for( int i = 1; i <= q; ++i) {
        int l = gn(), r = gn();
        ret = 0;
        query(root[r], 1, n, l);
    }
}
```

### 带修改莫队

```cpp
int a[N], sum[N], num[N * 10];
ll ans[N]; ll cnt = 0;

struct node {
    int l, r, pre, id;
}s[N];

struct star {
    int pos, id;
}op[N];

inline void add(int x) {
    cnt += num[sum[x]];
    num[sum[x]]++;
}

inline void sub(int x) {
    num[sum[x]]--;
    cnt -= num[sum[x]];
}

inline void work(int x, int l, int r) {
    x = op[x].pos;
    if(x >= l && x <= r) {
        sub(x);
        sum[x] = sum[x] ^ a[x] ^ a[x + 1];
        add(x);
        swap(a[x], a[x + 1]);
    } else {
        sum[x] = sum[x] ^ a[x] ^ a[x + 1];
        swap(a[x], a[x + 1]);
    }

}

int main() {
    int block = pow(n, 2.0 / 3.0);

    int Cnum = 0, Qnum = 0;
    for(int i = 1; i <= m; ++i) {
        int cmd = gn();
        if(cmd == 1) {
            s[++Qnum] = {gn() - 1, gn(), Cnum, Qnum};
        } else {
            op[++Cnum] = {gn(), Cnum};
        }
    }

    sort(s + 1, s + 1 + Qnum, [&](node a, node b) {
        if(a.l / block != b.l / block) return a.l / block < b.l / block;
        if(a.r / block != b.r / block) return a.r / block < b.r / block;
        return a.pre < b.pre;
    });

    int l = 1, r = 0, now = 0;
    for(int i = 1; i <= Qnum; ++i) {
        while(s[i].l < l) add(--l);
        while(s[i].r > r) add(++r);
        while(s[i].l > l) sub(l++);
        while(s[i].r < r) sub(r--);
        while(now < s[i].pre) work(++now, l, r);
        while(now > s[i].pre) work(now--, l, r);
        ans[s[i].id] = 1LL * (s[i].r - s[i].l) * (s[i].r - s[i].l + 1) / 2LL - cnt;
    }
}
```

### 线段树最大子段和

```cpp
struct star {
     ll lsum, rsum, sum, ans;
}s[N << 2];

void pushup(int node) {
    s[node].sum = s[lson].sum + s[rson].sum;
    s[node].lsum = max(s[lson].lsum, s[lson].sum + s[rson].lsum);
    s[node].rsum = max(s[rson].rsum, s[rson].sum + s[lson].rsum);
    s[node].ans = max(s[lson].rsum + s[rson].lsum, max(s[lson].ans, s[rson].ans));
}

void build(int node, int l, int r) {
    if(l == r) {
        s[node].lsum = s[node].rsum = s[node].sum = a[l];
        s[node].ans = a[l];
        return ;
    }
    int mid = l + r >> 1;
    build(lson, l, mid);
    build(rson, mid + 1, r);
    pushup(node);
}

star query(int node, int l, int r, int L, int R) {
    if(L <= l && R >= r) {
        return s[node];
    }
    int mid = l + r >> 1;
    ll sign = -1e9;
    star lp = {sign, sign, sign, sign}, rp = {sign, sign, sign, sign}, ans = {sign, sign, sign, sign};
    if(L <= mid) lp = query(lson, l, mid, L, R);
    if(R > mid) rp = query(rson, mid + 1, r, L, R);
    ans.sum = lp.sum + rp.sum;
    ans.lsum = max(lp.lsum, lp.sum + rp.lsum);
    ans.rsum = max(rp.rsum, rp.sum + lp.rsum);
    ans.ans = max(lp.rsum + rp.lsum, max(lp.ans, rp.ans));
    return ans;

}
```

### 莫队+ST表

```cpp
constexpr int mod = 1e9 + 7;
constexpr int N = 1e5 + 5;

//莫队算法
int a[N], ans[N], pos[N], len, mp[N];

struct node {
    int l, r, k;
}s[N];

int v[N];
int f[N][25], dp[N][25], n, m, LOG[N], k;

inline where(int x) {
    return lower_bound(v, v + k, x) - v + 1;
}

inline void add(int node) {
    if(!mp[a[node]]) ++len;
    mp[a[node]]++;
}

inline void sub(int node) {
    mp[a[node]]--;
    if(!mp[a[node]]) --len;
}

inline void ST_prework() {
    for(int i = 1; i <= n; ++i) {
        f[i][0] = dp[i][0] = a[i];
        LOG[i] = log2(i);
    }
    int t = LOG[n] + 1;
    for(int j = 1; j < t; ++j) {
        for(int i = 1; i <= n - (1 << j) + 1; ++i) {
            f[i][j] = max(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);
            dp[i][j] = min(dp[i][j - 1], dp[i + (1 << (j - 1))][j - 1]);
        }
    }
}

inline int ST_query(int l, int r) {
    int k = LOG[r - l + 1];
    return max(f[l][k], f[r - (1 << k) + 1][k]) - min(dp[l][k], dp[r - (1 << k) + 1][k]) + 1;
}

int main() {
      len = 0;
      n = gn(), m = gn();
      int block = sqrt(n);
      for(int i = 1; i <= n; ++i) {
          mp[i] = 0;
          a[i] = gn();
          pos[i] = i / block;
          v[i - 1] = a[i];
      }
      ST_prework();

      sort(v, v + n);
      k = unique(v, v + n) - v;
      for(int i = 1; i <= n; ++i) a[i] = where(a[i]);
      for(int i = 1; i <= m; ++i) {
          s[i].l = gn(), s[i].r = gn(), s[i].k = i;
      }
      sort(s + 1, s + 1 + m, [](node a, node b) {
          if(pos[a.l] == pos[b.l]){
                if(pos[a.l] % 2) return  a.r < b.r;
                return a.r > b.r;
            }
            return pos[a.l] < pos[b.l];
      });
      int l = 1, r = 0;
      for(int i = 1; i <= m; ++i) {
          while(s[i].l < l) add(--l);
          while(s[i].r > r) add(++r);
          while(s[i].l > l) sub(l++);
          while(s[i].r < r) sub(r--);
          if(ST_query(s[i].l, s[i].r) == len) {
              ans[s[i].k] = 1;
          } else ans[s[i].k] = 0;
      }
}
```

### 线段树左右区间前缀维护

```cpp
//lp表示最左边的端点 rp表示最右端的端点 lsum左侧最大和 rsum右侧最大和

struct node{
    int lsum, rsum, lp, rp, sum;
}s[N<<2];
void pushup(int node,int l,int r){
    int mid=(l+r)>>1;

    ////左右端点
    s[node].lp=s[lson].lp;
    s[node].rp=s[rson].rp;

    ////合并sum
    if(s[lson].rp==s[rson].lp){
        s[node].sum=max(s[lson].sum,s[rson].sum);
    } else{
        s[node].sum=(s[lson].rsum+s[rson].lsum);
        s[node].sum=max(s[node].sum,s[lson].sum);
        s[node].sum=max(s[node].sum,s[rson].sum);
    }

    if(s[lson].rp!=s[rson].lp&&s[lson].lsum==(mid-l+1)){
        s[node].lsum=(s[lson].lsum+s[rson].lsum);
    }else {
        s[node].lsum=s[lson].lsum;
    }

    if(s[lson].rp!=s[rson].lp&&s[rson].rsum==(r-mid)){
        s[node].rsum=(s[lson].rsum+s[rson].rsum);
    }else {
        s[node].rsum=s[rson].rsum;
    }
}

void build(int node,int l,int r){
    if(l==r){
        s[node].lsum=s[node].rsum=s[node].sum=1;
        s[node].lp=s[node].rp=0;
        return ;
    }
    int mid=l+r>>1;
    build(lson,l,mid);
    build(rson,mid+1,r);
    pushup(node,l,r);
}

void change(int node,int l,int r,int idx){
    if(l==r){
        s[node].lsum=s[node].rsum=s[node].sum=1;
        s[node].lp=!s[node].lp;
        s[node].rp=!s[node].rp;
        return ;
    }
    int mid=l+r>>1;
    if(idx<=mid) change(lson,l,mid,idx);
    else change(rson,mid+1,r,idx);
    pushup(node,l,r);
}
```

### 树链剖分+线段树

```jsx
int n,m,r,p;
vector<int> v[N];
int dep[N],f[N],siz[N],son[N],top[N];
int id[N],tot=0,a[N],out[N],mp[N];
void dfs(int node,int fa){
    dep[node]=dep[fa]+1;
    f[node]=fa;
    siz[node]=1;
    int maxn=0;
    for(int k:v[node]){
        if(k==fa)continue;
        dfs(k,node);
        siz[node]+=siz[k];
        if(siz[k]>maxn){
            maxn=siz[k];
            son[node]=k;
        }
    }
}
void dfs1(int node,int topx){
    top[node]=topx;
    id[node]=++tot;
    mp[tot]=node;
    if(son[node]){
        dfs1(son[node],topx);
    }
    for(int k:v[node]){
        if(k==f[node]||k==son[node])continue;
        dfs1(k,k);
    }
    out[node]=tot;
}
ll t[N<<2],lazy[N<<2];
void pushup(int node){
    t[node]=(t[lson]+t[rson])%p;
}
void build(int node,int l,int r){
    if(l==r){
        t[node]=a[mp[l]];
        return ;
    }
    int mid=l+r>>1;
    build(lson,l,mid);
    build(rson,mid+1,r);
    pushup(node);
}
void spread(int node,int l,int r){
    if(lazy[node]){
        int mid=l+r>>1;
        t[lson]=(t[lson]+lazy[node]*(mid-l+1))%p;
        t[rson]=(t[rson]+lazy[node]*(r-mid))%p;
        lazy[lson]=(lazy[lson]+lazy[node])%p;
        lazy[rson]=(lazy[rson]+lazy[node])%p;
        lazy[node]=0;
    }
}
void change(int node,int l,int r,int L,int R,int val){
    if(L<=l&&R>=r){
        lazy[node]=(lazy[node]+val)%p;
        t[node]=(t[node]+val*(r-l+1))%p;
        return ;
    }
    spread(node,l,r);
    int mid=l+r>>1;
    if(L<=mid) change(lson,l,mid,L,R,val);
    if(R>mid) change(rson,mid+1,r,L,R,val);
    pushup(node);
}
ll query(int node,int l,int r,int L,int R){
    if(L<=l&&R>=r){
        return t[node];
    }
    spread(node,l,r);
    int mid=l+r>>1;
    ll val=0;
    if(L<=mid) val=(val+query(lson,l,mid,L,R))%p;
    if(R>mid) val=(val+query(rson,mid+1,r,L,R))%p;
    return val%p;
}
int main(){
    n=gn(),m=gn(),r=gn(),p=gn();
    repi(i,1,n){
        a[i]=gn();
        if(a[i]>p)a[i]%=p;
    }
    repi(i,2,n){
        int x=gn(),y=gn();
        v[x].pb(y);
        v[y].pb(x);
    }
    dfs(r,0);
    dfs1(r,r);
    build(1,1,n);
    repi(i,1,m){
        int cmd=gn();
        if(cmd==1){
            int x=gn(),y=gn(),k=gn();
            while(top[x]!=top[y]){
                if(dep[top[x]]>=dep[top[y]]){
                    change(1,1,n,id[top[x]],id[x],k);
                    x=f[top[x]];
                }else {
                    change(1,1,n,id[top[y]],id[y],k);
                    y=f[top[y]];
                }
            }
            int l=min(id[x],id[y]),r=max(id[x],id[y]);
            change(1,1,n,l,r,k);
        }else if(cmd==2){
            int x=gn(),y=gn();
            ll ans=0;
            while(top[x]!=top[y]){
                if(dep[top[x]]>=dep[top[y]]){
                    ans+=query(1,1,n,id[top[x]],id[x]);
                    if(ans>p)ans%=p;
                    x=f[top[x]];
                }else {
                    ans+=query(1,1,n,id[top[y]],id[y]);
                    if(ans>p)ans%=p;
                    y=f[top[y]];
                }
            }
            int l=min(id[x],id[y]),r=max(id[x],id[y]);
            ans+=query(1,1,n,l,r);
            printf("%lld\n",ans%p);
        }else if(cmd==3){
            int x=gn(),k=gn();
            change(1,1,n,id[x],out[x],k);
        }else {
            int x=gn();
            printf("%lld\n",query(1,1,n,id[x],out[x]));
        }
    }
}
```

### 扫描线

```jsx
struct star{
    ll x, y, h, val;
}t[N];
int tot = 0;
vector<int> v;
int where(int x) {
    return lower_bound(v.begin(), v.end(), x) - v.begin() + 1;
}
struct node {
    ll sum,val,len;
}s[N<<2];
void pushup(int node, int l, int r) {
    if(s[node].sum) {
        s[node].val = s[node].len;
    }else s[node].val = s[lson].val + s[rson].val;
}
void build(int node, int l, int r) {
    if(l == r) {
        s[node].len = v[l] - v[l-1];
        return ;
    }
    int mid = (l + r) >> 1;
    build(lson, l, mid);
    build(rson, mid + 1, r);
    s[node].len = s[lson].len + s[rson].len;
}
void change(int node, int l, int r, int L, int R, int val) {
    if(L <= l && R >= r){
        s[node].sum += val;
        pushup(node, l, r);
        return ;
    }
    int mid = (l+r) >> 1;
    if(L <= mid) change(lson, l, mid, L, R, val);
    if(R > mid) change(rson, mid + 1, r, L, R, val);
    pushup(node, l, r);
}
int main(){
    int n = gn();
    for(int i = 1; i <= n; ++i) {
        int x=gn(),y=gn(),_x=gn(),_y=gn();
        t[++tot] = {x, _x, y, 1};
        t[++tot] = {x, _x, _y, -1};
        v.pb(x),v.pb(_x);
    }
    sort(v.begin(), v.end());
    v.erase(unique(v.begin(),v.end()),v.end());
    int len = v.size();
    build(1, 1, len - 1);

    ll ans=0;
    sort(t + 1, t + 1 + tot, [](star a,star b){
        if(a.h==b.h) return a.val>b.val;
        return a.h<b.h;
    });
    for(int i = 1; i <= tot - 1; ++i) {
        change(1,1,len-1,where(t[i].x),where(t[i].y)-1,t[i].val);
        ans+=s[1].val*(t[i+1].h-t[i].h);
    }

    print(ans);
    putchar(10);
}
```

### 主席树

```jsx
int n, m;
vector <int> v;
struct node {
    int lc, rc, sum;
}s[N * 40];

int tot = 0, root[N], a[N];
void insert(int l, int r, int pre, int &now, int idx) {
    s[++tot] = s[pre];
    now = tot;
    s[now].sum++;
    if(l == r) return ;
    int mid = l + r >> 1;
    if(idx <= mid) insert(l, mid, s[pre].lc, s[now].lc, idx);
    else insert(mid + 1, r, s[pre].rc, s[now].rc, idx);
}

int query(int l, int r, int L, int R, int k) {
    if(l == r) return l;
    int mid = l + r >> 1;
    int tem = s[s[R].lc].sum - s[s[L].lc].sum;
    if(k <= tem) return query(l, mid, s[L].lc, s[R].lc, k);
    else return query(mid + 1, r, s[L].rc, s[R].rc, k - tem);
}

int where(int x) {
    return lower_bound(all(v), x) - v.begin() + 1;
}
int main(){
    n = gn(), m = gn();
    repi(i, 1, n) {
        a[i] = gn();
        v.pb(a[i]);
    }
    sort(all(v));
    v.erase(unique(all(v)), v.end());
    int len = v.size();
    repi(i, 1, n) {
        insert(1, len, root[i - 1], root[i], where(a[i]));
        //cout << where(a[i]) << endl;
    }
    repi(i, 1, m) {
        int l = gn(), r = gn(), k = gn();
        print(v[query(1, len, root[l - 1], root[r], k) - 1]);
        putchar(10);
    }
}
```

## 奇奇怪怪的东西

1. 贪心如果拿不定可以两种情况取小 二次排序注意$id$
2.  树上差分统计每条边经过次数 不同方向不同差分数组 下标不要用错
3. 让所有点两两联通 $\rightarrow$ 最小生成树 超源超汇思想
4. 补图最小生成树  set维护当前还不在最小生成树上的点集。或者根号分治
5. 超源求所有点到某一点的最短路
6. 推公式 然后数据结构维护 把相同元素放到等号一边
7. 带权中位数 效仿普通中位数 线段树树上二分维护
8. 当某个限定值比较小的时候 建立多颗线段树
9. 拓扑转移 注意单调性
10. 注意题干中的细节条件 尤其范围 别读错题了0.0
11. 莫队离线可以维护很牛逼的东西
12. 数据随机 有很好的性质 比如暴力下放不会被卡（下放的次数不会太多） 如果离线之后按权值排序，可以将区间修改转化为区间赋值
13. $2$倍转化为$log$
14. 线段树/主席树维护哈希
15. 离线+带撤销并查集
16. 权值线段树离线求$mex$ 在线则主席树
17. 如果发现答案很容易$judge$的时候，可以考虑二分 二分思想
18. 