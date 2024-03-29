# 数据结构

[TOC]



### 基础数据结构

#### 单调栈求以某个数为最大（最小）值的区间范围

```cpp
// 此为单调递减栈 单调队列同理 灵活应用
void solve(int l, int r) {
    int top = 0;
    st[top] = l - 1;
    for(int i = l; i <= r; ++i) {
        while(top and a[i] > a[st[top]]) {
            L[st[top]] = st[top - 1] + 1;
            R[st[top]] = i - 1;
            top--;
        }
        st[++top] = i;
    }
    while(top) {
        L[st[top]] = st[top - 1] + 1;
        R[st[top]] = r;
        top--;
    }
    ll ans = 0;
    for(int i = l; i <= r; ++i) {
        ans = ans + 1ll * (i - L[i] + 1) * (R[i] - i + 1) * a[i];
    }
    cout << ans << endl;
}
```

```cpp
// 此为单调递减栈 单调队列同理 灵活应用
void solve(int l, int r) {
    int top = 0;
    st[top] = l - 1;
    for(int i = l; i <= r; ++i) {
        while(top and a[i] > a[st[top]]) {
            L[st[top]] = st[top - 1] + 1;
            R[st[top]] = i - 1;
            top--;
        }
        st[++top] = i;
    }
    while(top) {
        L[st[top]] = st[top - 1] + 1;
        R[st[top]] = r;
        top--;
    }
    ll ans = 0;
    for(int i = l; i <= r; ++i) {
        ans = ans + 1ll * (i - L[i] + 1) * (R[i] - i + 1) * a[i];
    }
    c
        out << ans << endl;
}
```



### 并查集

#### 扩展域并查集

```cpp
// from 1 to n express good, from n + 1 to n + n express bad
struct node {
    int x, y;
    long long w;

    bool operator<(const node &rhs) const {
        return w > rhs.w;
    }
} s[N];

int f[M * 2];// 二倍空间

void init(int n) {
    for (int i = 1; i <= n; ++i) {
        f[i] = i; f[n + i] = n + i;
    }
}

int found(int x) {
    if (f[x] == x) return x;
    return f[x] = found(f[x]);
}

bool isunit(int x, int y) {
    x = found(x);
    y = found(y);
    if (x == y) return true;
    return false;
}

void unit(int x, int y) {
    x = found(x);
    y = found(y);
    f[x] = y;
}

int n, m;

int main() {
    for (int i = 1; i <= m; ++i) {
        if (s[i].x == s[i].y)continue;
        if (!isunit(s[i].x, s[i].y)) {
            unit(s[i].x, s[i].y + n);
            unit(s[i].x + n, s[i].y);
        }
    }
}
```

#### 带撤销并查集

```cpp
/*
给出一个n个点m条边的无向联通图(n,m<=5e5)，有q(q<=5e5)个询问
每个询问询问一个边集{Ei}，回答这些边能否在同一个最小生成树中

要知道一个性质，就是权值不同的边之间是独立的，即权值为x的所有边的选取不影响权值>x的边的选取
于是我们可以把所有询问离线，按边权排序，对于当前处理的边权，如果有某个询问在其中，那么我们把这些边加进去看有没有环，如果有，那么这个询问就被叉掉了，当然处理完了还要把刚才的操作撤销掉
处理了当前权值x的所有询问，最后别忘了把权值为x的边做kruskal算法加进去
这样时间复杂度是带log的（按秩合并的可撤销并查集的复杂度）
 */
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



### 块状数据结构

### 堆

#### 堆+贪心求第K大

```cpp
/*
求区间长度在[L, R]之间的第K大区间和
*/
constexpr int N = 5e5 + 100;

ll a[N];
ll sum[N];

struct ST_table {
    ll f[N][30], LOG[N];
    ll id[N][30];

    void ST_work(int n) {
        for(int i = 1; i <= n; ++i) {
            f[i][0] = sum[i];
            id[i][0] = i;
            LOG[i] = log2(i);
        }

        int t = LOG[n] + 1;
        for(int j = 1; j < t; ++j) {
            for(int i = 1; i <= n - (1 << j) + 1; ++i) {
                if(f[i][j - 1] < f[i + (1 << (j - 1))][j - 1]) {
                    f[i][j] = f[i + (1 << (j - 1))][j - 1];
                    id[i][j] = id[i + (1 << (j - 1))][j - 1];
                } else {
                    f[i][j] = f[i][j - 1];
                    id[i][j] = id[i][j - 1];
                }
            }
        }
    }

    pair<ll, ll> ST_query(int l, int r) {
        int k = LOG[r - l + 1];
        if(f[l][k] < f[r - (1 << k) + 1][k]) {
            return {f[r - (1 << k) + 1][k], id[r - (1 << k) + 1][k]};
        }
        return {f[l][k], id[l][k]};
    }
}st;

struct node {
    ll l, r, _l, _r, val;
    bool operator <  (const node &rhs) const {
        return val < rhs.val;
    }
};

int main() {

    int n = gl(), k = gl(), L = gl(), R = gl();
    for(int i = 1; i <= n; ++i) {
        a[i] = gl();
        sum[i] = sum[i - 1] + a[i];
    }

    st.ST_work(n);

    priority_queue<node> q;
    for(int i = 1; i + L - 1 <= n; ++i) {
        ll l = i + L - 1, r = min(i + R - 1, n);
        pair<ll, ll> now = st.ST_query(l, r);
        q.push({i, now.second, l, r, now.first - sum[i - 1]});
    }

    ll ans = 0;

    for(int i = 1; i <= k; ++i) {
        node star = q.top();
        q.pop();
        ans += star.val;
        ll l = star._l, r = star.r - 1;
        pair<ll, ll> k{-INF, 0};
        if(l <= r) {
            k = st.ST_query(l, r);
            q.push({star.l, k.second, l, r, k.first - sum[star.l - 1]});
        }
        l = star.r + 1, r = star._r;
        if(l <= r) {
            k = st.ST_query(l, r);
            q.push({star.l, k.second, l, r, k.first - sum[star.l - 1]});
        }
    }

    cout << ans << endl;
}
```



### ST表

```cpp
int f[N][25], n, m, LOG[N], k;

inline void ST_prework() {
    for(int i = 1; i <= n; ++i) {
        f[i][0] = a[i];
        LOG[i] = log2(i);
    }
    int t = LOG[n] + 1;
    for(int j = 1; j < t; ++j) {
        for(int i = 1; i <= n - (1 << j) + 1; ++i) {
            f[i][j] = max(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);
        }
    }
}

inline int ST_query(int l, int r) {
    int k = LOG[r - l + 1];
    return max(f[l][k], f[r - (1 << k) + 1][k]);
}
```



### 莫队

#### 普通莫队

```cpp
constexpr int N = 1e5 + 5;
int a[N], ans[N], pos[N], len, mp[N], n, m;

struct node {
    int l, r, k;
}s[N];

inline void add(int node) {
    if(!mp[a[node]]) ++len;
    mp[a[node]]++;
}

inline void sub(int node) {
    mp[a[node]]--;
    if(!mp[a[node]]) --len;
}

int main() {
      len = 0;
      n = gn(), m = gn();
      int block = sqrt(n);
      for(int i = 1; i <= n; ++i) {
          mp[i] = 0; a[i] = gn();
          pos[i] = i / block;
      }
    
      for(int i = 1; i <= m; ++i) {
          s[i].l = gn(), s[i].r = gn(), s[i].k = i;
      }
      sort(s + 1, s + 1 + m, [](node a, node b) {
          if(pos[a.l] == pos[b.l]){
                if(pos[a.l] % 2) return  a.r < b.r;
                return a.r > b.r;
            }
            return a.l < b.l;
      });
      int l = 1, r = 0;
      for(int i = 1; i <= m; ++i) {
          while(s[i].l < l) add(--l);
          while(s[i].r > r) add(++r);
          while(s[i].l > l) sub(l++);
          while(s[i].r < r) sub(r--);
      }
}
```

#### 带修改莫队

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
    
    // 奇偶优化排序
    sort(s + 1, s + 1 + Qnum, [&](node a, node b) {
        if(a.l / block != b.l / block) return a.l / block < b.l / block;
        if(a.r / block != b.r / block) {
            if (a.l / block & 1) return a.r / block < b.r / block;
            else return a.r / block > b.r / block;
        }
        if ((a.l / block & 1) == (a.r / block & 1)) return a.pre < b.pre;
        return a.pre > b.pre;
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

#### 莫队二次离线

```cpp
// 子区间 or 区间对数一类满足区间减法的问题
vector<int> buc;

vector<tuple<int, int, int> >  v[N];

ll cnt[N], pre[N];
ll ans[N];

struct node {
    int l, r, id;
    ll ans;
}q[N];

int main() {
    int n = gn(), m = gn(), k = gn();
    for (int i = 1; i <= n; ++i) a[i] = gn();

    for (int i = 1; i <= m; ++i) {
        int l = gn(), r = gn();
        q[i] = {l, r, i};
    }

    for (int i = 0, len = (1 << 14); i < len; ++i) {
        if (__builtin_popcount(i) == k) buc.push_back(i);
    }

    for (int i = 1; i <= n; ++i) {
        for (auto x : buc) cnt[a[i] ^ x] ++;
        pre[i] = cnt[a[i + 1]];
    }

    memset(cnt, 0, sizeof cnt);

    int block = sqrt(n);
    sort(q + 1, q + 1 + m, [&](node a, node b) {
        if (a.l / block != b.l / block) return a.l < b.l;
        return a.r < b.r;
    });

    // [l, r]
    for (int i = 1, l = 1, r = 0; i <= m; ++i) {
        // [l, r] -> [l + i, r]
        if (l < q[i].l) v[r].emplace_back(l, q[i].l - 1, -i);
        while (l < q[i].l) {
            q[i].ans += pre[l - 1];
            ++l;
        }

        // [l, r] -> [l - i, r]
        if (l > q[i].l) v[r].emplace_back(q[i].l, l - 1, i);
        while (l > q[i].l) {
            --l;
            q[i].ans -= pre[l - 1];
        }

        // [l, r] -> [l, r + i]
        if (r < q[i].r) v[l - 1].emplace_back(r + 1, q[i].r, -i);
        while (r < q[i].r) {
            q[i].ans += pre[r];
            ++r;
        }

        // [l, r] -> [l, r - i]
        if (r > q[i].r) v[l - 1].emplace_back(q[i].r + 1, r, i);
        while (r > q[i].r) {
            --r;
            q[i].ans -= pre[r];
        }
    }

    for (int i = 1; i <= n; ++i) {
        for (auto to : buc) cnt[a[i] ^ to] ++;
        for (auto x : v[i]) {
            auto[nl, nr, id] = x;
            for (int j = nl; j <= nr; ++j) {
                ll tem = cnt[a[j]];
                if (j <= i and k == 0) --tem;
                q[abs(id)].ans += ((id > 0) ? 1 : -1) * tem;
            }
        }
    }

    for (int i = 1; i <= m; ++i) q[i].ans += q[i - 1].ans;

    for (int i = 1; i <= m; ++i) ans[q[i].id] = q[i].ans;
}
```



### 树状数组

#### 树状数组维护前缀最大值

```cpp
struct FenwickTree {
    static const int M = 1e5 + 100;
    int maxn = 65536;
    int tr[M];

    int lowbit(int x) {
        return -x & x;
    }

    void add(int x, int val) {
        while(x <= maxn) {
            tr[x] = max(tr[x], val);
            x += lowbit(x);
        }
    }

    void clear(int x) {
        while(x <= maxn) {
            tr[x] = 0;
            x += lowbit(x);
        }
    }

    int query(int x) {
        int ans = 0;
        while(x) {
            ans = max(ans, tr[x]);
            x -= lowbit(x);
        }
        return ans;
    }
} tree;
```



### 线段树

#### 基础模板

```cpp
struct Segment {
    static const int N = 2e5 + 100;
    #define lson node << 1
    #define rson node << 1 | 1

    struct node {
        int lazy, sum;
    } s[N * 4];

    void spread(int node, int l, int r) {
        if(s[node].lazy) {
            int mid = l + r >> 1;
            s[lson].lazy += s[node].lazy;
            s[rson].lazy += s[node].lazy;
            s[lson].sum += s[node].lazy * (mid - l + 1);
            s[rson].sum += s[node].lazy * (r - mid);
            s[node].lazy = 0;
        }
    }

    void pushup(int node, int l, int r) {
        s[node].sum = s[lson].sum + s[rson].sum;
    }

    void build(int node, int l, int r, int *ar) {
        s[node].lazy = 0;
        if(l == r) {
            s[node].sum = ar[l];
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
            s[node].lazy += val;
            return ;
        }
        spread(node, l, r);
        int mid = l + r >> 1;
        if(L <= mid) change(lson, l, mid, L, R, val);
        if(R > mid) change(rson, mid + 1, r, L, R, val);
        pushup(node, l, r);
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
}s;
```



#### 最大连续1的个数

```cpp
//lp表示最左边的端点 rp表示最右端的端点 lsum左侧最大和 rsum右侧最大和

struct node {
    int lsum, rsum, lp, rp, sum;
} s[N << 2];

void pushup(int node, int l, int r) {
    int mid = (l + r) >> 1;

    ////左右端点
    s[node].lp = s[lson].lp;
    s[node].rp = s[rson].rp;

    ////合并sum
    if (s[lson].rp == s[rson].lp) {
        s[node].sum = max(s[lson].sum, s[rson].sum);
    } else {
        s[node].sum = (s[lson].rsum + s[rson].lsum);
        s[node].sum = max(s[node].sum, s[lson].sum);
        s[node].sum = max(s[node].sum, s[rson].sum);
    }

    if (s[lson].rp != s[rson].lp && s[lson].lsum == (mid - l + 1)) {
        s[node].lsum = (s[lson].lsum + s[rson].lsum);
    } else {
        s[node].lsum = s[lson].lsum;
    }

    if (s[lson].rp != s[rson].lp && s[rson].rsum == (r - mid)) {
        s[node].rsum = (s[lson].rsum + s[rson].rsum);
    } else {
        s[node].rsum = s[rson].rsum;
    }
}

void build(int node, int l, int r) {
    if (l == r) {
        s[node].lsum = s[node].rsum = s[node].sum = 1;
        s[node].lp = s[node].rp = 0;
        return;
    }
    int mid = l + r >> 1;
    build(lson, l, mid);
    build(rson, mid + 1, r);
    pushup(node, l, r);
}

void change(int node, int l, int r, int idx) {
    if (l == r) {
        s[node].lsum = s[node].rsum = s[node].sum = 1;
        s[node].lp = !s[node].lp;
        s[node].rp = !s[node].rp;
        return;
    }
    int mid = l + r >> 1;
    if (idx <= mid) change(lson, l, mid, idx);
    else change(rson, mid + 1, r, idx);
    pushup(node, l, r);
}
```

#### 最大子段和

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

#### 带权中位数

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

#### 离线维护区间mex（在线用主席树即可）

```cpp
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

#### 线段树合并

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

##### 例题

```cpp
/*
给定一棵树，每个点有颜色 C，多次查询，每次给定 u,v,l,r，你需要给出一个颜色 x使得x满足：
1. x∈ [l, r]
2. x 在 u 到 v 的路径上出现了奇数次。
*/
int n, m; int a[N]; ll rd[N];

mt19937 rnd(233); 
struct SegmentTree {
    static const int N = 3e5 + 100;
#define lson(x) s[x].lc
#define rson(x) s[x].rc

    struct node {
        int lc, rc;
        ll sum;
    } s[N * 80];

    int tot = 0;

    void insert(int &now, int l, int r, int id, int val) {
        if (not now) now = ++tot;
        s[now].sum ^= rd[id];
        if (l == r) return;
        int mid = (l + r) >> 1;
        if (id <= mid) insert(lson(now), l, mid, id, val);
        else insert(rson(now), mid + 1, r, id, val);
    }

    int getid(int st, int ed, int fa, int gfa, int l, int r) {
        if (l == r) {
            if (s[st].sum ^ s[ed].sum ^ s[fa].sum ^ s[gfa].sum) return l;
            return -1;
        }
        int mid = (l + r) >> 1;
        ll num = s[lson(st)].sum ^ s[lson(ed)].sum ^ s[lson(fa)].sum ^ s[lson(gfa)].sum;
        if (num != 0) return getid(lson(st), lson(ed), lson(fa), lson(gfa), l, mid);

        num = s[rson(st)].sum ^ s[rson(ed)].sum ^ s[rson(fa)].sum ^ s[rson(gfa)].sum;
        if (num != 0) return getid(rson(st), rson(ed), rson(fa), rson(gfa), mid + 1, r);

        return -1;
    }

    int query(int st, int ed, int fa, int gfa, int l, int r, int L, int R) {
        if (l == L and R == r) {
            return getid(st, ed, fa, gfa, l, r);
        }
        int mid = (l + r) >> 1;
        if (R <= mid) return query(lson(st), lson(ed), lson(fa), lson(gfa), l, mid, L, R);
        else if (L > mid) return query(rson(st), rson(ed), rson(fa), rson(gfa), mid + 1, r, L, R);
        else {
            int num = query(lson(st), lson(ed), lson(fa), lson(gfa), l, mid, L, mid);
            if (num != -1) return num;
            else return query(rson(st), rson(ed), rson(fa), rson(gfa), mid + 1, r, mid + 1, R);
        }
    }

    void merge(int &u, int v) {
        if (not u or not v) {
            u += v;
            return ;
        }

        s[u].sum ^= s[v].sum;

        merge(lson(u), lson(v));
        merge(rson(u), rson(v));
    }

} tree;

int root[N]; int dep[N], dp[N][25];
vector<int> v[N];
void dfs(int node, int fa) {
    dep[node] = dep[fa] + 1;
    dp[node][0] = fa;
    for (int i = 1; (1 << i) <= dep[node]; ++i) {
        dp[node][i] = dp[dp[node][i - 1]][i - 1];
    }
    for (auto to : v[node]) {
        if (to == fa) continue;
        tree.merge(root[to], root[node]);
        dfs(to, node);
    }
}

int lca(int x, int y) {
    if (dep[x] < dep[y]) swap(x, y);
    int tem = dep[x] - dep[y];
    for (int i = 0; tem; ++i) {
        if (tem & 1) x = dp[x][i];
        tem >>= 1;
    }
    if (x == y) return x;
    for (int j = 21; j >= 0 and x != y; --j) {
        if (dp[x][j] != dp[y][j]) {
            x = dp[x][j];
            y = dp[y][j];
        }
    }
    return dp[x][0];
}

int main() {
    n = gn(), m = gn();
    for (int i = 1; i <= n; ++i) {
        rd[i] = rnd();
    }
    for (int i = 1; i <= n; ++i) {
        a[i] = gn();
        tree.insert(root[i], 1, n, a[i], 1);
    }

    for (int i = 1; i < n; ++i) {
        int x = gn(), y = gn();
        v[x].push_back(y);
        v[y].push_back(x);
    }

    dfs(1, 0);

    for (int i = 1; i <= m; ++i) {
        int u = gn(), t = gn(), l = gn(), r = gn();
        int fa = lca(u, t);
        int num = tree.query(root[u], root[t], root[fa], root[dp[fa][0]], 1, n, l, r);
        cout << num << endl;
    }

}
```

##### 线段树合并+CDQ分治统计答案

```cpp
/*
给一颗以1为根的树。
每个点有两个权值：vi, ti,一开始全部是零。
Q次操作：
读入o, u, d
o = 1 对u到根上所有点的vi += d 
o = 2 对u到根上所有点的ti += vi * d
最后,输出每个点的ti值(n, Q <= 100000)
*/
const int N = 1e5 + 100;

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



### 二维线段树

#### 区间查询+单点修改

```cpp
struct SegmentTree {
    static const int M = 1050;
 #define lson node << 1
 #define rson node << 1 | 1
    // in tree begin
    struct IT {
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

        void change(int node, int l, int r, int idx, int val) {
            if(l == r) {
                maxn[node] = minx[node] = val;
                return ;
            }
            int mid = l + r >> 1;
            if(idx <= mid) change(lson, l, mid, idx, val);
            else change(rson, mid + 1, r, idx, val);
            pushup(node, l, r);
        }

        pair<int, int> query(int node, int l, int r, int L, int R) {
            if(L == l && R == r) {
                return {maxn[node], minx[node]};
            }
            int mid = l + r >> 1;
            if(R <= mid) return query(lson, l, mid, L, R);
            else if(L > mid) return query(rson, mid + 1, r, L, R);
            else {
                pair<int, int> lf, rt;
                lf = query(lson, l, mid, L, mid);
                rt = query(rson, mid + 1, r, mid + 1, R);
                return {max(lf.first, rt.first), min(lf.second, rt.second)};
            }
        }

    };

    IT tree[M << 2];

    void pushup(int node, int l, int r, IT &now, IT &lf, IT &rt) {
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

    void update(int node, int l, int r, int idx, IT &now, IT &lf, IT &rt) {
        now.maxn[node] = max(lf.maxn[node], rt.maxn[node]);
        now.minx[node] = min(lf.minx[node], rt.minx[node]);
        if(l == r) return ;
        int mid = l + r >> 1;
        if(idx <= mid) update(lson, l, mid, idx, now, lf, rt);
        else update(rson, mid + 1, r, idx, now, lf, rt);
    }
    
    void change(int node, int l, int r, int idy, int X, int Y, int val) {
        if(l == r) {
            tree[node].change(1, 1, idy, Y, val);
            return ;
        }
        int mid = l + r >> 1;
        if(X <= mid) change(lson, l, mid, idy, X, Y, val);
        else change(rson, mid + 1, r, idy, X, Y, val);
        update(1, 1, idy, Y, tree[node], tree[lson], tree[rson]);
    }

    pair<int, int> query(int node, int l, int r, int idy, int xL, int xR, int yL, int yR) {
        if(xL == l && xR == r) {
            return tree[node].query(1, 1, idy, yL, yR);
        }
        int mid = l + r >> 1;
        if(xR <= mid) return query(lson, l, mid, idy, xL, xR, yL, yR);
        else if(xL > mid) return query(rson, mid + 1, r, idy, xL, xR, yL, yR);
        else {
            pair<int, int> lf, rt;
            lf = query(lson, l, mid, idy, xL, mid, yL, yR);
            rt = query(rson, mid + 1, r, idy, mid + 1, xR, yL, yR);
            return {max(lf.first, rt.first), min(lf.second, rt.second)};
        }
    }
} tr;
```

### 二维树状数组

#### 单点修改+区间查询

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

#### 区间修改+单点查询

二维前缀和：$sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + a[i][j]$

我们可以令差分数组 $d[i][j]$ 表示$a[i][j]$ 与 $a[i - 1][j] + a[i][j - 1] - a[i - 1][j - 1]$的差

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



### 扫描线

```cpp
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
}s[N << 2];
void pushup(int node, int l, int r) {
    if(s[node].sum) s[node].val = s[node].len;
    else s[node].val = s[lson].val + s[rson].val;
}
void build(int node, int l, int r) {
    if(l == r) {
        s[node].len = v[l] - v[l - 1];
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
        int x = gn(), y = gn(), _x = gn(), _y = gn();
        t[++tot] = {x, _x, y, 1};
        t[++tot] = {x, _x, _y, -1};
        v.push_back(x),v.push_back(_x);
    }
    sort(v.begin(), v.end());
    v.erase(unique(v.begin(), v.end()), v.end());
    int len = v.size();
    build(1, 1, len - 1);

    ll ans = 0;
    sort(t + 1, t + 1 + tot, [](star a, star b){
        if(a.h == b.h) return a.val > b.val;
        return a.h < b.h;
    });
    for(int i = 1; i <= tot - 1; ++i) {
        change(1, 1, len - 1, where(t[i].x), where(t[i].y) - 1, t[i].val);
        ans += s[1].val * (t[i + 1].h - t[i].h);
    }

    cout << ans << '\n';
}
```

### 线段树维护树上连通性

```cpp
/*
给出一个 n 个点的树。点的权值是一个 0 − n − 1 的排列。支持以下 2种操作：
▶ 交换点 i 和 j 的权值
▶ 询问若从树上找一条简单路径，则其上的所有点的权值的集合的Mex 值最大是多少
思路：
线段树区间 [l,r] 维护权值为 l-r 的点是否在一条简单路径上，若是，同时维护满足条件的最短路径的两个端点。合并子树信息即是两条路径的并。二分答案 x，线段树查询 1-x 是否在一条路径上
*/
int dis(int x, int y) {
    return dep[x] + dep[y] - 2 * dep[lca(x, y)];
}

struct SegmentTree {
    static const int maxn = 2e5 + 100;
    pair<int, int> s[maxn << 2];
    #define lson node << 1
    #define rson node << 1 | 1

    function<pair<int, int> (pair<int, int>, int)> merge = [&](pair<int, int> point, int c) -> pair<int, int> {
        if (point.first == -1 or c == -1) return {-1, -1};
        int a = point.first;
        int b = point.second;
        int a_to_b = dis(a, b);
        int a_to_c = dis(a, c);
        int b_to_c = dis(b, c);
        if (a_to_b + a_to_c == b_to_c) return {b, c};
        if (a_to_b + b_to_c == a_to_c) return {a, c};
        if (b_to_c + a_to_c == a_to_b) return {a, b};
        return {-1, -1};
    };

    void pushup(int node) {
        pair<int, int> lp = s[lson], rp = s[rson], x;
        x = merge(lp, rp.first);
        x = merge(x, rp.second);
        s[node] = x;
    }

    void build(int node, int l, int r) {
        if (l == r) {
            s[node] = {a[l], a[l]};
            return ;
        }
        int mid = l + r >> 1;
        build(lson, l, mid);
        build(rson, mid + 1, r);
        pushup(node);
    }

    void change(int node, int l, int r, int id, int val) {
        if (l == r) {
            s[node] = {val, val};
            return;
        }
        int mid = l + r >> 1;
        if (id <= mid) change(lson, l, mid, id, val);
        else change(rson, mid + 1, r, id, val);
        pushup(node);
    }

    int query(int node, int l, int r, pair<int, int> pre) {
        if (l == r) {
            pair<int, int> lp = pre, rp = s[node], x;
            x = merge(lp, rp.first);
            x = merge(x, rp.second);
            return x.first == -1 ? l - 1 : l;
        }
        int mid = l + r >> 1;
        pair<int, int> x = merge(pre, s[lson].first);
        x = merge(x, s[lson].second);
        if (x.first == -1) return query(lson, l, mid, pre);
        else return query(rson, mid + 1, r, x);
    }
} tree;

int main() {
    ios::sync_with_stdio(false);
    cin >> n;
    for (int i = 1; i <= n; ++i) {
        head[i] = -1;
        cin >> b[i]; b[i]++;
        a[b[i]] = i;
    }
    for (int i = 2; i <= n; ++i) {
        int x; cin >> x; add(x, i);
    }

    dfs(1, 0);
    dfs1(1, 1);

    tree.build(1, 1, n);

    int q; cin >> q;

    while (q--) {
        int cmd; cin >> cmd;
        if (cmd == 1) {
            int x, y; cin >> x >> y;
            swap(b[x], b[y]);
            a[b[x]] = x;
            a[b[y]] = y;
            tree.change(1, 1, n, b[x], a[b[x]]);
            tree.change(1, 1, n, b[y], a[b[y]]);
        } else {
            cout << tree.query(1, 1, n, {a[1], a[1]}) << endl;
        }
    }
}
```



### 线段树优化建图

```cpp
struct node {
    int to, net, w;
}s[N * 4];

int tot = 0, head[N * 4];

void add (int x, int y, int w) {
    s[++tot] = {y, head[x], w};
    head[x] = tot;
}

int val[N];

int cnt, ls[N], rs[N];

void build(int &p, int &q, int l, int r) {
    if (l == r) {
        p = q = l;
        return;
    }
    if (not p) p = ++cnt;
    if (not q) q = ++cnt;
    int mid = (l + r) >> 1;
    build(ls[p], ls[q], l, mid); add(p, ls[p], 0), add(ls[q], q, 0);
    build(rs[p], rs[q], mid + 1, r); add(p, rs[p], 0), add(rs[q], q, 0);
}

void change(int node, int l, int r, int L, int R, int rt, int flag) {
    if (L == l and R == r) {
        if (flag) add(node, rt, 0);
        else add(rt, node, 0);
        return;
    }
    int mid = l + r >> 1;
    if (R <= mid) change(ls[node], l, mid, L, R, rt, flag);
    else if (L > mid) change(rs[node], mid + 1, r, L, R, rt, flag);
    else {
        change(ls[node], l, mid, L, mid, rt, flag);
        change(rs[node], mid + 1, r, mid + 1, R, rt, flag);
    }
}


int main() {
    memset(head, -1, sizeof head);
    memset(val, 0x3f, sizeof val);
    n = gn(), m = gn(), k = gn();
    cnt = n;
    int rootp = 0, rootq = 0;
    build(rootp, rootq, 1, n);
    for (int i = 1; i <= m; ++i) {
        int l = ++cnt, r = ++cnt;
        int a = gn(), b = gn(), c = gn(), d = gn();
        add(l, r, 1);
        change(rootq, 1, n, a, b, l, 1);
        change(rootp, 1, n, c, d, r, 0);
        l = ++cnt, r = ++cnt;
        add(l, r, 1);
        change(rootq, 1, n, c, d, l, 1);
        change(rootp, 1, n, a, b, r, 0);
    }

    priority_queue<pair<int, int> > q;
    q.push({0, k});
    val[k] = 0;

    while (not q.empty()) {
        int now = q.top().second, num = -q.top().first;
        q.pop();
        if (num > val[now]) continue;
        for (int i = head[now]; ~i; i = s[i].net) {
            int to = s[i].to;
            if (val[to] > val[now] + s[i].w) {
                val[to] = val[now] + s[i].w;
                q.push({-val[to], to});
            }
        }
    }
}
```



### 李超线段树
```cpp
#include<bits/stdc++.h>

using namespace std;
const int MAXN = 1000005;

#define ll long long

struct Line {
    ll k, b;
    ll val (ll x) const { return k * x + b; }
    explicit Line(ll _k = 0, ll _b = 0) { k = _k; b = _b; }
};

const ll INF = (ll) 1e18;

struct LiChao_Segmenttree {
#define lson node << 1
#define rson node << 1 | 1
    static ll cross_x(const Line &A, const Line &B) { // 求交点
        return (B.b - A.b) / (A.k - B.k);
    }

    struct tree_node {
        bool vis;
        bool has_line;
        Line line;
    } s[MAXN << 2];

    void build(int node, int l, int r) {
        s[node].vis = false;
        s[node].has_line = false;
        if (l == r) return ;
        int mid = (l + r) >> 1;
        build(lson, l, mid);
        build(rson, mid + 1, r);
    }

    void insert(int node, int l, int r, int L, int R, Line x) {
        s[node].vis = true;
        if (l == L && r == R) {
            if (not s[node].has_line) {
                s[node].has_line = true;
                s[node].line = x;
                return;
            }
            if (s[node].line.val(l) >= x.val(l) and s[node].line.val(r) >= x.val(r)) return; // 原线段完全大于插入线段
            if (s[node].line.val(l) < x.val(l) and s[node].line.val(r) < x.val(r)) { s[node].line = x; return; } // 原线段完全小于插入线段
            int mid = (l + r) >> 1;
            if (cross_x(x, s[node].line) <= mid) { // 交点在 mid 左边
                if (x.k < s[node].line.k) insert(lson, l, mid, L, mid, x);
                else insert(lson, l, mid, L, mid, s[node].line), s[node].line = x;
            } else {
                if (x.k > s[node].line.k) insert(rson, mid + 1, r, mid + 1, R, x);
                else insert(rson, mid + 1, r, mid + 1, R, s[node].line), s[node].line = x;
            }
            return ;
        }

        int mid = (l + r) >> 1;
        if (r <= mid) insert(lson, l, mid, L, R, x);
        else if (l > mid) insert(rson, mid + 1, r, L, R, x);
        else {
            insert(lson, l, mid, L, mid, x);
            insert(rson, mid + 1, r, mid + 1, R, x);
        }
    }

    void clear(int node, int l, int r) {
        s[node].vis = false; s[node].has_line = false;
        if (l != r) {
            int mid = (l + r) >> 1;
            if (s[lson].vis) clear(lson, l, mid);
            if (s[rson].vis) clear(rson, mid + 1, r);
        }
    }

    ll get_val(int node, int l, int r, int x) {
        if (not s[node].vis) return -INF;
        ll ret;
        if (not s[node].has_line) ret = -INF;
        else ret = s[node].line.val(x);
        if (l == r) return ret;
        int mid = (l + r) >> 1;
        if (x <= mid) return max(ret, get_val(lson, l, mid, x));
        else return max(ret, get_val(rson, mid + 1, r, x));
    }
} seg_up, seg_down;


int main() {
    ios::sync_with_stdio(false);
    int n, m; cin >> n >> m;
    seg_up.build(1, 1, n);
    seg_down.build(1, 1, n);
    while (m--) {
        int op; cin >> op;
        if (op == 0) {
            int k, b; cin >> k >> b;
            seg_up.insert(1, 1, n, 1, n, Line(k, b));
            seg_down.insert(1, 1, n, 1, n, Line(-k, -b));
        } else {
            int x; cin >> x;
            printf("%lld %lld\n", seg_up.get_val(1, 1, n, x), -seg_down.get_val(1, 1, n, x));
        }
    }
    return 0;
}
```
### 树链剖分

```cpp
vector<int> v[N];

int dep[N], f[N], siz[N], son[N], top[N];
int id[N], tot = 0;

void predfs(int node, int fa) {
    dep[node] = dep[fa] + 1;
    siz[node] = 1;
    f[node] = fa;
    int maxn = 0;
    for (auto to : v[node]) {
        if (to == fa) continue;
        predfs(to, node);
        siz[node] += siz[to];
        if (siz[to] > maxn) {
            maxn = siz[to];
            son[node] = to;
        }
    }
}

void dfs(int node, int topx) {
    top[node] = topx;
    id [node] = ++tot;
    if (son[node]) dfs(son[node], topx);
    for (auto to : v[node]) {
        if (to == f[node] or to == son[node]) continue;
        dfs(to, to);
    }
}

int lca(int x, int y) {
    while (top[x] != top[y]) {
        if (dep[top[x]] >= dep[top[y]]) x = f[top[x]];
        else y = f[top[y]];
    }
    return dep[x] < dep[y] ? x : y;
}

int dis(int x, int y) {
    return dep[x] + dep[y] - 2 * dep[lca(x, y)];
}

struct SegmentTree {
#define lson node << 1
#define rson node << 1 | 1
    int num[N << 2], lazy[N << 2];

    void spread(int node) {
        if (lazy[node]) {
            num[lson] += lazy[node];
            num[rson] += lazy[node];
            lazy[lson] += lazy[node];
            lazy[rson] += lazy[node];
            lazy[node] = 0;
        }
    }

    void pushup(int node) {
        num[node] = max(num[lson], num[rson]);
    }

    void build(int node, int l, int r) {
        num[node] = lazy[node] = 0;
        if (l == r) return;
        int mid = l + r >> 1;
        build(lson, l, mid);
        build(rson, mid + 1, r);
    }

    void change(int node, int l, int r, int L, int R, int val) {
        if (L <= l and R >= r) {
            num[node] += val;
            lazy[node] += val;
            return;
        }
        int mid = l + r >> 1;
        spread(node);
        if (L <= mid) change(lson, l, mid, L, R, val);
        if (R > mid) change(rson, mid + 1, r, L, R, val);
        pushup(node);
    }

    int query(int node, int l, int r, int L, int R) {
        if (L == l and R == r) {
            return num[node];
        }
        int mid = l + r >> 1;
        spread(node);
        if (R <= mid) return query(lson, l, mid, L, R);
        else if (L > mid) return query(rson, mid + 1, r, L, R);
        else {
            int val;
            val += query(lson, l, mid, L, mid);
            val += query(rson, mid + 1, r, mid + 1, R);
            return val;
        }
    }
} tree;

struct node {
    int fi, st, dis;
} road[N];

int main() {
    int n = gn(), m = gn();
    for (int i = 1; i < n; ++i) {
        int x = gn(), y =gn();
        v[y].emplace_back(x);
        v[x].emplace_back(y);
    }

    predfs(1, 0);
    dfs(1, 1);

    tree.build(1, 1, n);

    for (int i = 1; i <= m; ++i) {
        int x = gn(), y = gn();
        if (id[x] > id[y]) swap(x, y);
        road[i] = {x, y, dis(x, y)};
    }

    for (int i = 1; i <= m; ++i) {
        // query
        int x = road[i].fi, y = road[i].st;

        pair<int, int> now, star;
        while(top[x] != top[y]) {
            if(dep[top[x]] >= dep[top[y]]) {
                tree.query(1, 1, n, id[top[x]], id[x]);
                x = f[top[x]];
            }else {
                tree.query(1, 1, n, id[top[y]], id[y]);
                y = f[top[y]];
            }
        }
        int l = min(id[x], id[y]), r = max(id[x], id[y]);
        tree.query(1, 1, n, l, r);
        // add
        x = road[i].fi, y = road[i].st;

        while(top[x] != top[y]) {
            if(dep[top[x]] >= dep[top[y]]) {
                tree.change(1, 1, n, id[top[x]], id[x], 1);
                x = f[top[x]];
            }else {
                tree.change(1, 1, n, id[top[y]], id[y], 1);
                y = f[top[y]];
            }
        }
        tree.change(1, 1, n, l, r, 1);
    }
}
```



### 吉司机线段树

吉司机线段树是一种势能线段树，可以实现区间取 $min/max$(给定 $l,r,x$ 把所有满足 $l≤i≤r$ 的 $a_i$ 改成 $min(a_i,x)$ 和区间求和

以 $min$为例，线段树上每个节点维护四个值：

- $mx$：区间最大值
- $cnt$：区间最大值的出现次数
- $md$：区间次大值（严格小于最大值且最大的数）
- $sum$：区间和

实现区间取 $min$ 时，递归到线段树上一个包含于询问区间的节点 $p$ 时，进行如下处理：

- 若 $ x≥mx_p$，则显然这次修改不影响节点 $p$，直接 $return$
- 若 $x≤md_p$，则暴力往 $p$ 的左右子节点递归
- 否则 $md_p<x<mx_p$，这次修改对 $sum_p$ 可以计算出为 $cnt_p \times (mx_p−x)$，打标记即可

```cpp
#pragma GCC optimize(3) // 手动开O^3

struct SegmentBeats {
    static const int maxn = 2e5 + 100;

    #define lson node << 1
    #define rson node << 1 | 1

    struct node {
        int minx, seminx, cnt, lazy, sum;
        int num[32];
    }s[maxn << 2];

    __attribute__((optimize("O3"),target("avx"))) // 卡常
    void pushup(int node, int l, int r) {
        for (int i = 0; i <= 30; ++i) {
            s[node].num[i] = s[lson].num[i] + s[rson].num[i];
        }

        s[node].minx = s[lson].minx, s[node].cnt = s[lson].cnt, s[node].seminx = s[lson].seminx;

        if(s[rson].minx < s[node].minx) {
            s[node].minx = s[rson].minx, s[node].cnt = s[rson].cnt;
            s[node].seminx = min(s[lson].minx, s[rson].seminx);
        } else if (s[rson].minx == s[node].minx) {
            s[node].cnt += s[rson].cnt;
            s[node].seminx = min(s[node].seminx, s[rson].seminx);
        } else {
            s[node].seminx = min(s[node].seminx, s[rson].minx);
        }

        s[node].sum = s[lson].sum ^ s[rson].sum;
    }

    __attribute__((optimize("O3"),target("avx")))
    void spread(int node, int l, int r) {
        if(s[node].lazy > s[lson].minx and s[node].lazy < s[lson].seminx) {
            for (int i = 0; i <= 30; ++i) {
                if(s[lson].minx & (1 << i)) s[lson].num[i] -= s[lson].cnt;
            }
            if(s[lson].cnt & 1) s[lson].sum ^= s[lson].minx;

            s[lson].minx = s[node].lazy;

            for (int i = 0; i <= 30; ++i) {
                if(s[lson].minx & (1 << i)) s[lson].num[i] += s[lson].cnt;
            }
            if(s[lson].cnt & 1) s[lson].sum ^= s[lson].minx;
            s[lson].lazy = max(s[node].lazy, s[lson].lazy);
        }
        if(s[node].lazy > s[rson].minx and s[node].lazy < s[rson].seminx) {
            for (int i = 0; i <= 30; ++i) {
                if(s[rson].minx & (1 << i)) s[rson].num[i] -= s[rson].cnt;
            }
            if(s[rson].cnt & 1) s[rson].sum ^= s[rson].minx;

            s[rson].minx = s[node].lazy;

            for (int i = 0; i <= 30; ++i) {
                if(s[rson].minx & (1 << i)) s[rson].num[i] += s[rson].cnt;
            }
            if(s[rson].cnt & 1) s[rson].sum ^= s[rson].minx;
            s[rson].lazy = max(s[node].lazy, s[rson].lazy);
        }
        s[node].lazy = 0;
    }

    __attribute__((optimize("O3"),target("avx")))
    void build (int node, int l, int r) {
        if(l == r) {
            s[node].minx = a[l], s[node].seminx = (1 << 30);
            s[node].cnt = 1; s[node].lazy = 0; s[node].sum = a[l];
            for (int i = 0; i <= 30; ++i) {
                if (a[l] & (1 << i)) s[node].num[i] = 1;
                else s[node].num[i] = 0;
            }
            return ;
        }
        int mid = l + r >> 1;
        build(lson, l, mid);
        build(rson, mid + 1, r);
        pushup(node, l, r);
    }

    __attribute__((optimize("O3"),target("avx")))
    void change(int node, int l, int r, int L, int R, int x) {
        if(L == l and R == r) {
            if(x <= s[node].minx) return ;
            else if (x < s[node].seminx) {
                for (int i = 0; i <= 30; ++i) {
                    if(s[node].minx & (1 << i)) s[node].num[i] -= s[node].cnt;
                }
                if(s[node].cnt & 1) s[node].sum ^= s[node].minx;

                s[node].minx = x;

                for (int i = 0; i <= 30; ++i) {
                    if(s[node].minx & (1 << i)) s[node].num[i] += s[node].cnt;
                }
                if(s[node].cnt & 1) s[node].sum ^= s[node].minx;
                s[node].lazy = x;
                return ;
            }
        }
        if(s[node].lazy) spread(node, l, r);
        int mid = l + r >> 1;
        if(R <= mid) change(lson, l, mid, L, R, x);
        else if (L > mid) change(rson, mid + 1, r, L, R, x);
        else change(lson, l, mid, L, mid, x), change(rson, mid + 1, r, mid + 1, R, x);
        pushup(node, l, r);
    }

    __attribute__((optimize("O3"),target("avx")))
    int query (int node, int l, int r, int L, int R, int maxbit) {
        if(L <= l and R >= r) {
            return s[node].num[maxbit];
        }
        if(s[node].lazy) spread(node, l, r);
        int mid = l + r >> 1;
        int ans = 0;
        if(L <= mid) ans += query(lson, l, mid, L, R, maxbit);
        if(R > mid) ans += query(rson, mid + 1, r, L, R, maxbit);
        return ans;
    }

    __attribute__((optimize("O3"),target("avx")))
    int querysum(int node, int l, int r, int L, int R) {
        if(L <= l and R >= r) {
            return s[node].sum;
        }
        if(s[node].lazy) spread(node, l, r);
        int mid = l + r >> 1;
        int ans = 0;
        if(L <= mid) ans ^= querysum(lson, l, mid, L, R);
        if(R > mid) ans ^= querysum(rson, mid + 1, r, L, R);
        return ans;
    }
}tree;
```



### 平衡树

#### Splay

##### 基础板子

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

##### 区间翻转

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

##### 区间处理备用

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

#### fhq Treap

##### fhq Treap 平衡树基本操作（按值分裂）

```cpp
mt19937 rnd(time(0));

struct fhqTreap{
    #define l(x) fhq[x].l
    #define r(x) fhq[x].r 
    struct Node {
        int l, r;
        int val, rd;
        int siz;
    }fhq[N];

    int cnt = 0, root = 0;

    inline int newnode(int val) {
        fhq[++cnt].val = val;
        fhq[cnt].rd = rnd();
        fhq[cnt].siz = 1;
        return cnt;
    }

    inline void update(int now) {
        fhq[now].siz = fhq[l(now)].siz + fhq[r(now)].siz + 1;
    }

    void split(int now, int val, int &x, int &y) {
        if (not now) {
            x = y = 0;
            return ;
        }
        if (fhq[now].val <= val) {
            x = now;
            split(fhq[now].r, val, fhq[now].r, y);
        } else {
            y = now;
            split(fhq[now].l, val, x, fhq[now].l);
        }
        update(now);
    }

    int merge(int x, int y) {
        if (not x or not y) return x + y;
        if (fhq[x].rd > fhq[y].rd) {
            fhq[x].r = merge(fhq[x].r, y);
            update(x);
            return x;
        } else { 
            fhq[y].l = merge(x, fhq[y].l);
            update(y);
            return y;
        }
    }

    int x, y, z;
    inline void ins(int val) {
        split(root, val, x, y);
        root = merge(merge(x, newnode(val)), y);
    }

    inline void del(int val) {
        split(root, val, x, z);
        split(x, val - 1, x, y);
        y = merge(l(y), r(y));
        root = merge(merge(x, y), z);
    }

    inline void getrank(int val, int &rank) {
        split(root, val - 1, x, y);
        rank = fhq[x].siz + 1;
        root = merge(x, y);
    }

    inline void getnum(int rank, int &val) {
        int now = root;
        while (now) {
            if (fhq[l(now)].siz + 1 == rank) break;
            else if (fhq[l(now)].siz >= rank) now = l(now);
            else {
                rank -= fhq[l(now)].siz + 1;
                now = r(now);
            }
        }
        val = fhq[now].val;
    }

    inline void pre(int val, int &id) {
        split(root, val - 1, x, y);
        int now = x;
        while (r(now)) now = r(now);
        id = fhq[now].val;
        root = merge(x, y);
    } 

    inline void nxt(int val, int &id) {
        split(root, val, x, y);
        int now = y;
        while (l(now)) now = l(now);
        id = fhq[now].val;
        root = merge(x, y);
    } 
    
} tree;
```

##### fhq Treap 区间操作（按大小分裂）

```cpp
mt19937 rnd(233);
struct fhqTreap {
    #define l(x) fhq[x].l
    #define r(x) fhq[x].r
    #define rd(x) fhq[x].rd
    #define val(x) fhq[x].val
    #define siz(x) fhq[x].siz
    #define rev(x) fhq[x].rev

    struct node {
        int l, r, val, rd, siz;
        bool rev;
    }fhq[N];

    int cnt = 0, root = 0;

    inline void update (int now) {
        siz(now) = siz(l(now)) + siz(r(now)) + 1;
    }

    inline int newnode (int val) {
        ++cnt;
        fhq[cnt] = {0, 0, val, (int)rnd(), 1};
        return cnt;
    }

    inline void spread(int now) {
        swap(l(now), r(now));
        rev(l(now)) ^= 1;
        rev(r(now)) ^= 1;
        rev(now) = 0;
    }

    inline void split(int now, int siz, int &x, int &y) {
        if (not now) {
            x = y = 0;
            return ;
        }
        if (rev(now)) spread(now);
        if (siz(l(now)) < siz) {
            x = now;
            split(r(now), siz - siz(l(now)) - 1, r(now), y);
        } else {
            y = now;
            split(l(now), siz, x, l(now));
        }
        update(now);
    }

    inline int merge(int x, int y) {
        if (not x or not y) return x + y;
        if (rd(x) > rd(y)) {
            if (rev(x)) spread(x);
            r(x) = merge(r(x), y);
            update(x);
            return x;
        } else {
            if (rev(y)) spread(y);
            l(y) = merge(x, l(y));
            update(y);
            return y;
        }
    }

    void reverse(int l, int r) {
        int x, y, z;
        split(root, l - 1, x, y);
        split(y, r - l + 1, y, z);
        rev(y) ^= 1;
        root = merge(merge(x, y), z);
    }
}tree;
```



### PBDS

```cpp
#include <bits/stdc++.h>
#include <bits/extc++.h>
#include <ext/pb_ds/tree_policy.hpp> // tree
#include <ext/pb_ds/hash_policy.hpp> // hash
#include <ext/pb_ds/trie_policy.hpp> // trie
#include <ext/pb_ds/priority_queue.hpp> // priority_queue

using namespace __gnu_pbds;
using namespace std;

// hash

cc_hash_table<int, bool> hone; // 拉链法
gp_hash_table<int, bool> htwo; // 探测法

// 探测法会稍微快一些 用法跟map一样 复杂度比map优秀

// tree 不支持插入重复元素

#define pii pair<int, int>

tree<pii, null_type, less<pii>, rb_tree_tag, tree_order_statistics_node_update> tr, b;

/*
pii 存储类型
null_type 无映射 低版本g++为null_mapped_type
less<pii> 从小到大排序 greater<pii> 从大到小排序 or cmp
rb_tree_tag 红黑树
tree_order_statistics_node_update 更新方式
*/

// priority_queue

__gnu_pbds::priority_queue<int, greater<int>, pairing_heap_tag> Q, q;
/*
pairing_heap_tag push/join 为O(1) 其余均摊为log(n)
*/
int main() {

    int x = 2, y = 3;

    tr.insert({x, y}); // 插入
    tr.erase({x, y}); // 删除
    tr.order_of_key({x, y}); // 比当前元素小的个数
    tr.find_by_order(x); // 找第x + 1小的值
    tr.join(b); // 将b树并入tr 保证没有重复元素
    tr.split({x, y}, b); // 分裂 key小于等于v的元素属于tr 其余属于b
    tr.lower_bound({x, y}); // >=
    tr.upper_bound({x, y}); // >

    Q.pop();
    Q.push(x);
    Q.top();
    Q.join(q); // 合并
    Q.split(Pred prd. priority_queue &other) //分裂
    Q.empty();
    Q.size();
    Q.modify(iterator, val); // 修改一个节点的值
    Q.erase(it);
    // 还可以用迭代器迭代
}
```



### 跳表

### 可持久化线段树

#### 基本模板

```cpp
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
    for (int i = 1; i <= n; ++i) {
        a[i] = gn();
        v.push_back(a[i]);
    }
    sort(v.begin(), v.end());
    v.erase(unique(v.begin(), v.end()), v.end());
    int len = v.size();
    for (int i = 1; i <= n; ++i) {
        insert(1, len, root[i - 1], root[i], where(a[i]));
    }
    for (int i = 1; i <= m; ++i) {
        int l = gn(), r = gn(), k = gn();
        cout << v[query(1, len, root[l - 1], root[r], k) - 1])] << '\n';
    }
}
```

#### 统计区间内不同数字的个数

```cpp
const int N = 3e4 + 100;

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



### 可持久化并查集

### 可持久化字典树

### 珂朵莉树套线段树

```cpp
/*
给出一个 1 至 n 的排列，支持以下 3 种操作：
▶ 将区间 [l,r] 中的元素从小到大排序。
▶ 将区间 [l,r] 中的元素从大到小排序。
▶ 询问区间 [l,r] 的元素之和。
*/
struct node {
    int l, r;
    bool operator < (const node &rhs) const {
        return l < rhs.l;
    }
};

int sign[N];

struct SegmentTree {
    static const int maxn = 1e5 + 100;
    #define lson(x) s[x].lc
    #define rson(x) s[x].rc
    struct node {
        int lc, rc;
        int num;
    } s[maxn * 80];

    int root[maxn];
    int tot = 0;

    void insert(int &rt, int l, int r, int idx, int num) {
        if(!rt) rt = ++tot;

        s[rt].num += num;

        if(l == r) return ;

        int mid = (l + r) >> 1;

        if(idx <= mid) insert(lson(rt), l, mid, idx, num);
        else insert(rson(rt), mid + 1, r, idx, num);
    }

    int query(int rt, int l, int r) {
        if(l == r) return l;
        int mid = (l + r) >> 1;
        return s[s[rt].lc].num ? query(s[rt].lc, l, mid) : query(s[rt].rc, mid + 1, r);
    }

    void merge(int &u, int v) {
        if(not u or not v) {
            u += v;
            return ;
        }

        s[u].num += s[v].num;

        merge(lson(u), lson(v));
        merge(rson(u), rson(v));
    }

    void split(int x, int &y, int k, bool flag) {
        y = ++tot;
        s[y].num = s[x].num - k;
        s[x].num = k;

        if (flag) {
            int num = s[s[x].lc].num;
            if (num < k) split(s[x].rc, s[y].rc, k - num, flag);
            else swap(s[x].rc, s[y].rc);
            if (num > k) split(s[x].lc, s[y].lc, k, flag);
        } else {
            int num = s[s[x].rc].num;
            if (num < k) split(s[x].lc, s[y].lc, k - num, flag);
            else swap(s[x].lc, s[y].lc);
            if (num > k) split(s[x].rc, s[y].rc, k, flag);
        }
    }

} tree;

set<node> st;

set<node>::iterator spilt(int pos) {
    auto to = st.lower_bound({pos});
    if (to != st.end() and to->l == pos) {
        return to;
    }
    --to;
    int l = to->l, r = to->r;
    st.erase(to);
    int root = 0;
    tree.split(tree.root[l], tree.root[pos], pos - l, sign[l]);
    sign[pos] = sign[l];
    st.insert({l, pos - 1});
    return  st.insert({pos, r}).first;
}

void assign(int l, int r, int flag) {
    auto itr = spilt(r + 1), itl = spilt(l);

    for (set<node>::iterator it = ++itl; it != itr; ++it) {
        tree.merge(tree.root[l], tree.root[it->l]);
    }
    st.erase(itl, itr);

    st.insert({l, r});

    sign[l] = flag;
}
```



### Link-Cut-Tree

### 树套树

#### 树状数组套主席树

```cpp
// Dynamic ChairmanTree

// tree is a normal ChairmanTree and query is ArrayTree add ChairmanTree

vector<int> v;

struct node {
    int l, r, x;
    int id, type;
}s[N];

int a[N], len;
int totone, tottwo, qone[N], qtwo[N];

struct ChairmanTree {
    static const int maxn = 2e5 + 7;
#define lson(x) s[x].lc
#define rson(x) s[x].rc

    struct node {
        int lc, rc, val;
    }s[maxn * 100];

    int tot = 0, root[maxn];

    void insert(int &now, int pre, int l, int r, int idx, int val) {
        s[++tot] = s[pre];
        now = tot;
        s[now].val += val;
        if(l == r) return ;
        int mid = l + r >> 1;
        if(idx <= mid) insert(lson(now), lson(pre), l, mid, idx, val);
        else insert(rson(now), rson(pre), mid + 1, r, idx, val);
    }

    int query(int L, int R, int l, int r, int k, ChairmanTree &tr) {
        if(l == r) return v[l - 1];
        int x = s[lson(R)].val - s[lson(L)].val;
        for(int i = 1; i <= totone; ++i) x -= tr.s[tr.s[qone[i]].lc].val;
        for(int i = 1; i <= tottwo; ++i) x += tr.s[tr.s[qtwo[i]].lc].val;
        int mid = l + r >> 1;
        if(x >= k) {
            for (int i = 1; i <= totone; ++i) qone[i] = tr.s[qone[i]].lc;
            for (int i = 1; i <= tottwo; ++i) qtwo[i] = tr.s[qtwo[i]].lc;
            return query(lson(L), lson(R), l, mid, k, tr);
        } else {
            for (int i = 1; i <= totone; ++i) qone[i] = tr.s[qone[i]].rc;
            for (int i = 1; i <= tottwo; ++i) qtwo[i] = tr.s[qtwo[i]].rc;
            return query(rson(L), rson(R), mid + 1, r, k - x, tr);
        }

    }

}tree, query;

int lowbit(int x) { return -x & x;}

int where(int x) {
    return lower_bound(v.begin(), v.end(), x) - v.begin() + 1;
}

int main() {
    int n = gn(), m = gn();

    for(int i = 1; i <= n; ++i) {
        a[i] = gn();
        v.push_back(a[i]);
    }

    for(int i = 1; i <= m; ++i) {
        char c;
        cin >> c;
        if(c == 'Q') {
            s[i] = {gn(), gn(), gn()};
            s[i].type = 1;
        } else {
            s[i].id = gn(), s[i].x = gn();
            v.push_back(s[i].x);
            s[i].type = 0;
        }
    }

    sort(v.begin(), v.end());
    v.erase(unique(v.begin(), v.end()), v.end());
    len = v.size();

    for(int i = 1; i <= n; ++i) {
        tree.insert(tree.root[i], tree.root[i - 1], 1, len, where(a[i]), 1);
    }

    for(int i = 1; i <= m; ++i) {
        if (s[i].type == 1) {
            totone = 0, tottwo = 0;
            for(int j = s[i].l - 1; j > 0; j -= lowbit(j)) qone[++totone] = query.root[j];
            for(int j = s[i].r; j > 0; j -= lowbit(j)) qtwo[++tottwo] = query.root[j];
            int l = tree.root[s[i].l - 1], r = tree.root[s[i].r];
            printf("%d\n", tree.query(l, r, 1, len, s[i].x, query));
        } else {
            int x = s[i].id;
            int pre = a[x];
            a[x] = s[i].x;
            while(x <= n) {
                int preroot = query.root[x];
                query.insert(query.root[x], preroot, 1, len, where(pre), -1);
                preroot = query.root[x];
                query.insert(query.root[x], preroot, 1, len, where(s[i].x), 1);
                x += lowbit(x);
            }
        }
    }
}
```



### CDQ分治解决复杂数据结构问题

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

### 点分治

#### 空间换时间

给一棵带边权的树，问全部路径中前 m 大的。

二分第 m 大的值，每次用点分治检验合法性。二分完了以后再跑一次点分统计答案。而后第一个二分的时候直接作是 $n\times logn^3$​的，考虑降下来一个 $log$ 。先$dfs$一次树把每一个点做为重心的时候的全部距离预处理下来就能够省掉一个 $log$

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
