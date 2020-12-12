## 数据结构

[TOC]
### 单调栈求以某个数为最大（最小）值的区间范围
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

### 字典树+贪心求两数异或最大值
```cpp
// codeforces 282E
// 给了一个长度为 n(1 ≤ n ≤ 1e5) 的数组，求一个不相交的前缀和后缀，使得这个前缀和后缀中的所有数的异或值最大

ll a[N];
ll val = 0;

array<int, 60> getnum (ll num) {
    array<int, 60> ans; ans.fill(0);

    int cnt = 50;
    while(num) {
        if(num & 1) ans[cnt] = 1;
        num >>= 1;
        cnt--;
    }

    return ans;
}

struct DictionaryTree {
    int tr[N][2];

    int tot = 0;

    void insert(ll num) {
        int root = 0;
        array<int, 60> ans = getnum(num);
        for(int i = 1; i <= 50; ++i) {
            if(!tr[root][ans[i]]) tr[root][ans[i]] = ++tot;
            root = tr[root][ans[i]];
        }
        return ;
    }

    ll getans(ll num) {
        int root = 0;
        array<int, 60> ans = getnum(num);
        ll sum = 0;
        for(int i = 1; i <= 50; ++i) {
            if(tr[root][ans[i] ^ 1]) {
                sum = sum + (1LL << (50 - i));
                root = tr[root][ans[i] ^ 1];
            } else root = tr[root][ans[i]];
        }
        return max(sum, num);
    }
}tree;

int main() {
   int n = gn();
   for(int i = 1; i <= n; ++i) {
       a[i] = gl(); val ^= a[i];
   }

   ll ans = 0, num = 0;
   for(int i = n; i >= 0; --i) {
       ans = max(ans, tree.getans(val));
       num ^= a[i];
       val ^= a[i];
       tree.insert(num);
   }
   cout << ans << endl;
}
```

### 树状数组维护前缀最大值

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

### 点分序列 + 优先队列配合ST表求前K大

```cpp
/*
 * 给定一个N个结点的树，结点用正整数1到N编号。
 * 每条边有一个正整数权值。用d(a,b）表示从结点a到结点b路边上经过边的权值。
 * 其中要求a<b.将这n*(n-1)/2个距离从大到小排序，输出前M个距离值。
 */

struct node {
    int to, net;
    ll w;
}s[M];
int head[M], id = 0, tot = 0;
ll l[N], r[N], val[N];

void add(int x, int y, ll w) {
    s[++id] = {y, head[x], w};
    head[x] = id;
}

int root = 0, siz[N], vis[N];

void dfs_rt(int node, int fa, ll siztot) {
    siz[node] = 1;
    ll maxn = 0;
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(to == fa or vis[to]) continue;
        dfs_rt(to, node, siztot);
        siz[node] += siz[to];
        if(siz[to] > maxn) maxn = siz[to];
    }
    maxn = max(maxn, siztot - siz[node]);
    if(maxn * 2 <= siztot) root = node;
}

void dfs_dis(int node, int fa, ll w) {
    ++tot;
    val[tot] = w; l[tot] = l[tot - 1];
    if(!r[tot]) r[tot] = r[tot - 1];
    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(to == fa or vis[to]) continue;
        dfs_dis(to, node, w + s[i].w);
    }
}

void divide(int node, ll siztot) {
    dfs_rt(node, 0, siztot);
    node = root;
    dfs_rt(node, 0, siztot);
    
    ++tot;
    l[tot] = tot, r[tot] = tot - 1; val[tot] = 0;
    vis[node] = 1;

    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(vis[to]) continue;
        r[tot + 1] = tot;
        dfs_dis(to, node, s[i].w);
    }

    for(int i = head[node]; ~i; i = s[i].net) {
        int to = s[i].to;
        if(vis[to]) continue;
        divide(to, siz[to]);
    }
}

struct ST_table {
    static const int M = 3e6 + 100;
    ll id[M][30], LOG[M];

    void ST_work(int n) {
        for (int i = 1; i <= n; ++i) {
            id[i][0] = i;
            if(i >= 2) LOG[i] = LOG[i / 2] + 1;
        }
        int t = LOG[n] + 1;
        for (int j = 1; j < t; ++j) {
            for (int i = 1; i <= n - (1 << j) + 1; ++i) {
                id[i][j] = id[i + (1 << (j - 1))][j - 1];
                if (val[id[i][j - 1]] > val[id[i][j]]) {
                    id[i][j] = id[i][j - 1];
                }
            }
        }
    }

    int query(ll l, ll r) {
        int k = LOG[r - l + 1];
        if(val[id[l][k]] > val[id[r - (1 << k) + 1][k]]) return id[l][k];
        return id[r - (1 << k) + 1][k];
    }
} stTable;

struct star {
    ll l, r, _l, _r, val;
    bool operator < (const star &rhs) const {
        return val < rhs.val;
    }
};

priority_queue<star> q;

int main() {
    int n = gn(), m = gn();

    for(int i = 1; i <= n; ++i) head[i] = -1;
    for(int i = 1; i < n; ++i) {
        int x = gn(), y = gn(), w = gn();
        add(x, y, w);
        add(y, x, w);
    }

    divide(1, n);
    stTable.ST_work(tot);

    for(int i = 1; i <= tot; ++i) {
        if(l[i] > r[i]) continue;
        int now = stTable.query(l[i], r[i]);
        q.push({i, now, l[i], r[i], val[i] + val[now]});
    }

    for(int i = 1; i <= m; ++i) {
        star now = q.top();
        q.pop();
        printf("%lld\n", now.val);
        int l, r;
        l = now._l, r = now.r - 1;
        if(l <= r) {
            int k = stTable.query(l, r);
            q.push({now.l, k, l, r, val[now.l] + val[k]});
        }

        l = now.r + 1, r = now._r;
        if(l <= r) {
            int k = stTable.query(l, r);
            q.push({now.l, k, l, r, val[now.l] + val[k]});
        }
    }
}
```

### 树状数组套主席树 动态第K大

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

### 平板电视 基础Splay

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



### 二维线段树（区间查询 + 单点修改）

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
const int 
