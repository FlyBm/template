## 其他

[TOC]
### java输入挂
```java
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
        while (true) {
            int h = nextInt();
            if (h == -1) break;
            int w = nextInt(), n = nextInt();
        }
    }
}
```
### __builtin函数
```cpp
__builtin_ffs(x)  //返回x中最后一个为1的位是从后向前的第几位
__builtin_popcount(x) // 1的个数
__builtin_ctz(x) // x末尾0的个数。x=0时结果未定义。
__builtin_clz(x) // x前导0的个数。x=0时结果未定义。
__builtin_parity(x) // x中1的奇偶性。
__builtin_popcountll(unsigned long long x);
```
### hash表
```cpp
typedef unsigned long long ull;
// 99991 , 3000017 , 7679977 , 19260817 , 7679977
struct HashMap{
    static const int mod = 12227;
    struct node {
        int nxt;
        ull w;
    }e[N];
    int etot = 0, head[mod], tot[N][2];
    void init () {
        etot = 0;
        memset(head, -1, sizeof head);
    }

    void add(int x, ull val) {
        e[++etot] = {head[x], val};
        head[x] = etot;
    }

    bool insert (ull hashVal, bool tp) {
        int x = hashVal % mod;
        for (int i = head[x]; ~i; i = e[i].nxt){
            if (hashVal == e[i].w){
                tot[i][tp]++;
                return true;
            }
        }
        add(x, hashVal);
        tot[etot][tp] = 1;
        tot[etot][tp ^ 1] = 0;
        return false;
    }

    bool check(){
        for (int i = 1; i <= etot; ++i) {
            if (tot[i][0]==tot[i][1] && tot[i][0] == 1) return true;
        }
        return false;
    }
}hashTable;
```
### 茅台随机数
```cpp
mt19937 rnd(time(0));
mt19937 rnd(chrono::system_clock::now().time_since_epoch().count());
int a[N];

void solve() {
    shuffle(a + 1, a + N, rnd);
    printf("%lld\n", default_random_engine(seed));
}
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
        //        获取输入
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
### Java读入优化
```java
import java.io.*;
import java.math.BigInteger;
import java.util.*;

public class Main {
    static class Scanner {
        BufferedReader br;
        StringTokenizer st;
        public Scanner(InputStream s) {
            br = new BufferedReader(new InputStreamReader(s));
        }

        public Scanner(FileReader f) {
            br = new BufferedReader(f);
        }

        public String next() throws IOException {
            while (st == null || !st.hasMoreTokens())
                st = new StringTokenizer(br.readLine());
            return st.nextToken();
        }

        public int nextInt() throws IOException {
            return Integer.parseInt(next());
        }

        public long nextLong() throws IOException {
            return Long.parseLong(next());
        }

        public double nextDouble() throws IOException {
            return Double.parseDouble(next());
        }

        public int[] nextIntArr(int n) throws IOException {
            int[] arr = new int[n];
            for (int i = 0; i < n; ++i) {
                arr[i] = Integer.parseInt(next());
            }
            return arr;
        }
    }

    public static void main(String[] args) throws IOException {
        PrintWriter pw = new PrintWriter(System.out);
        Scanner reader = new Scanner(System.in);
        pw.println("hello");
        pw.close();
    }
}
```
### 二、三分查找
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
