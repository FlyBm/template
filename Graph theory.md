# 图论

[TOC]



### 基本性质

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
- 我们可以在$O(n^2)$的时间内求出最小环的权值。枚举环的两个端点+最短路

### 矩阵树定理

### 二分图

#### 性质

关于二分图中其他的性质有：
二分图的最小顶点覆盖： 用最少的点让每条边都至少和其中一个点关联。 
$Knoig$定理：二分图的最小顶点覆盖数等于二分图的最大匹配数。

DAG图的最小路径覆盖：用尽量少的不相交简单路径覆盖有向无环图的所有顶点。 
引理：DAG图的最小路径覆盖数=节点数(n)-最大匹配数(m)

二分图的最大独立集 在Ｎ个点的图G中选出m个点，使这m个点两两之间没有边．求m最大值。 
引理：二分图的最大独立集数 = 节点数(n)—最大匹配数(m)

#### 判断二分图

```cpp
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
```

#### 二分图匹配

```cpp
// 无向图
bool sell(int x) {
    istrue[x] = true;
    for (auto to : edge[x]) {
        if (not istrue[to]) {
            istrue[to] = true;
            if (not fang[to] or sell(fang[to])) {
                fang[to] = x; fang[x] = to;
                return true;
            }
        }
    }
    return false;
}

int cnt = 0;
for (int i = 1; i <= n; ++i) {
    if (fang[i]) continue;
    for (int j = 1; j <= n; ++j) istrue[j] = false;
    if (sell(i)) ++cnt;
}

//匈牙利算法求最大匹配
bool sell(int x) {
    vis[x] = 1;
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



### 欧拉回路

对于欧拉回路的判断：
无向图G存在欧拉通路的充要条件是：
$G$为连通图，并且$G$仅有两个奇度结点（度数为奇数的顶点）或者无奇度结点。
推论1：

1. 当$G$是仅有两个奇度结点的连通图时，$G$的欧拉通路必以此两个结点为端点。
2. 当$G$是无奇度结点的连通图时，$G$必有欧拉回路。
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

**缩点求2-SAT**

```cpp
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



### 连通性问题

#### 强连通分量（有向图）

```cpp
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
```

#### 割点

```cpp
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
#### 点双连通分量
```cpp
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
#### 割边（桥）

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

#### 有向图缩点

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

#### 全家桶

```cpp
struct node{
    int u, v, next;
}edge[N];

int head[N], ip;

void init(){
    memset(head, -1, sizeof(head));
    ip = 1;
}

void addedge(int u, int v){
    edge[++ip] = {u, v, head[u]};
    head[u] = ip;
}

namespace e_dcc { // 桥和便双连通分量
    int Count = 0, sol = 0, dcc = 0, bigdcc = 0, cnt = 0;
    // sol桥的数量 dcc边双连通分量的个数
    bool bridge[N * 2];
    int low[N], dfn[N];
    int color[N];

    void dfs(int x) {
        ++cnt;
        color[x] = dcc;
        for (int i = head[x]; ~i; i = edge[i].next) {
            int to = edge[i].v;
            if (color[to] or bridge[i]) continue;
            dfs(to);
        }
    }

    void tarjan(int node, int in_edge) {
        dfn[node] = low[node] = ++Count;
        for(int i = head[node]; ~i; i = edge[i].next){
            int to = edge[i].v;
            if (not dfn[to]) {
                tarjan(to, i);
                low[node] = min(low[node], low[to]);

                if (low[to] > dfn[node]) {
                    bridge[i] = bridge[i ^ 1] = true;
                }
            } else if (i != (in_edge ^ 1)) low[node] = min(low[node], dfn[to]);
        }
    }

    map<int, int> mp;
    int maxnsiz = 1;
    void getans (int n, int m) {
        Count = 0, sol = 0, dcc = 0, bigdcc = 0;
        mp.clear(), maxnsiz = 1;
        for (int i = 1; i <= n; ++i) {
            dfn[i] = low[i] = color[i] = 0;
            Count = 0;
        }

        for (int i = 1; i <= 2 * m + 2; ++i) bridge[i] = false;

        for (int i = 1; i <= n; ++i) {
            if (not dfn[i]) tarjan(i, 0);
        }

        for (int i = 1; i <= n; i++){
            if (not color[i]) {
                ++dcc;
                cnt = 0;
                dfs(i);
                if (cnt > 1) ++bigdcc;
            }
        }

        for (int i = 2; i < ip; i += 2) {
            if (bridge[i]) ++sol;
            if (color[edge[i].u] == color[edge[i].v]) {
                int num = color[edge[i].u];
                mp[num] ++;
                maxnsiz = max(maxnsiz, mp[num]);
            }
        }
    }
}

namespace v_dcc { // 割点
    int dfn[N], low[N], Count = 0;
    int vis[N];

    set<int> ans;
    void tarjan(int node, int rt){
        dfn[node] = low[node] = ++Count;
        vis[node] = 1;
        int child = 0;
        for(int i = head[node]; i != -1; i = edge[i].next){
            int to = edge[i].v;
            if(!dfn[to]) {
                ++child;
                tarjan(to, rt);
                low[node] = min(low[to], low[node]);
                if(node != rt and low[to] >= dfn[node]) {
                    ans.insert(node);
                }
            } else if(vis[to]) {
                low[node] = min(low[node], dfn[to]);
            }
        }
        if(child >= 2 and node == rt) {
            ans.insert(node);
        }
    }

    void getans (int n) {
        ans.clear();
        for (int i = 1; i <= n; ++i) {
            dfn[i] = low[i] = vis[i] = 0;
            Count = 0;
        }

        for (int i = 1; i <= n; ++i) {
            if (not dfn[i]) tarjan(i, i);
        }

    }
}
```



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

### 最短路

#### Johnson全源最短路（适用于稀疏图）

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

### 最小生成树

#### 01MST

```cpp
/*
给你一张n个点的完全图，其中有m条边长度为1，其余全为0。问你这张图的最小生成树为多少。
思路：就是求补图的连通块数量减一，可以用set的count函数来建立补图
*/
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



### 最小环

### LGV引理

### 网络流

#### 最大流

**$Dinic$**

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
**$HLPP$**

```cpp
#include <bits/stdc++.h>

using namespace std;

using Vi = vector<int>;

template<size_t MAXN, size_t MAXM, class T = int>
struct HLPP {
#define rep(i, b, e) for (int i = (b); i < (e); i++)
#define sz(x)        int((x).size())
    const T INF = numeric_limits<T>::max();
    struct edge {
        int to, rev;
        T f;
    };

    int maxn, s, t;

    edge edges[2 * MAXM];
    int first_edge[MAXN + 1];
    int _cur_edge[MAXN]; int nxt[MAXN]; int lst[MAXN];
    T excess[MAXN]; int arc[MAXN];

    int gapNxt[2 * MAXN], gapPrv[2 * MAXN];

    int height[MAXN];
    int highest, highestGap, work;
    int q[2 * MAXM];
    vector<int> degs;

    HLPP(vector<int> degrees, int s, int t) {
        this->s = s;
        this->t = t;
        maxn = degrees.size();
        assert(maxn <= MAXN);
        int cnt = 0;

        for (int i = 0; i < maxn; ++i) {
            first_edge[i] = cnt;
            cnt += degrees[i];
        }

        first_edge[maxn] = cnt;
        copy(first_edge, first_edge + maxn + 1, _cur_edge);
        degs.swap(degrees);
    }

    void addEdge(int from, int to, int f, bool isDirected = true) {
        edges[_cur_edge[from]++] = {to, _cur_edge[to], f};
        edges[_cur_edge[to]++] = {from, _cur_edge[from] - 1, isDirected ? 0 : f};
    }

    void pushLst(int h, int v) {
        nxt[v] = lst[h];
        lst[h] = v;
    }

    void updHeight(int v, int nh) {
        if (height[v] != maxn) {
            gapNxt[gapPrv[v]] = gapNxt[v];
            gapPrv[gapNxt[v]] = gapPrv[v];
        }

        height[v] = nh;

        if (nh == maxn)
            return;

        highestGap = max(highestGap, nh);

        if (excess[v] > 0) {
            highest = max(highest, nh);
            pushLst(nh, v);
        }

        nh += maxn;
        gapNxt[v] = gapNxt[nh];
        gapPrv[v] = nh;
        gapNxt[nh] = v;
        gapPrv[gapNxt[v]] = v;
    }

    void globalRelabel() {
        work = 0;
        fill(height, height + maxn, maxn);
        fill(lst, lst + maxn, -1);
        iota(gapNxt, gapNxt + maxn, 0);
        iota(gapPrv, gapPrv + maxn, 0);
        height[t] = 0;
        q[0] = t;
        int sz = 1;

        for (size_t i = 0; i < sz; ++i) {
            int v = q[i];
            for (int ie = first_edge[v]; ie < first_edge[v + 1]; ++ie) {
                auto &e = edges[ie];
                if (height[e.to] == maxn && edges[e.rev].f > 0)
                    q[sz++] = e.to, updHeight(e.to, height[v] + 1);
            }
            highest = highestGap = height[v];
        }
    }

    void push(int v, edge &e) {
        T df = min(excess[v], e.f);
        if (df > 0) {
            if (excess[e.to] == 0) pushLst(height[e.to], e.to);
            e.f -= df, edges[e.rev].f += df;
            excess[v] -= df, excess[e.to] += df;
        }
    }

    void discharge(int v) {
        int nh = maxn;

        rep(i, arc[v], first_edge[v + 1]) {
            auto &e = edges[i];
            if (e.f > 0) {
                if (height[v] == height[e.to] + 1) {
                    push(v, e);
                    if (excess[v] <= 0) {
                        arc[v] = i;
                        return;
                    }
                } else nh = min(nh, height[e.to] + 1);
            }
        }

        rep(i, first_edge[v], arc[v]) {
            auto &e = edges[i];
            if (e.f > 0) {
                if (height[v] == height[e.to] + 1) {
                    push(v, e);
                    if (excess[v] <= 0) {
                        arc[v] = i;
                        return;
                    }
                } else
                    nh = min(nh, height[e.to] + 1);
            }
        }
        work++;
        if (gapNxt[gapNxt[height[v] + maxn]] != height[v] + maxn) {
            updHeight(v, nh);
        } else {
            int oldH = height[v];
            rep(h, oldH, highestGap + 1) {
                for (int i = gapNxt[h + maxn]; i < maxn; i = gapNxt[i]) {
                    height[i] = maxn;
                }
                gapNxt[h + maxn] = gapPrv[h + maxn] = h + maxn;
            }
            highestGap = oldH - 1;
        }
    }

    T calc() {
        for (int v = 0; v < maxn; ++v) {
            sort(edges + first_edge[v], edges + first_edge[v + 1], [](edge &l, edge &r) {
                return l.to < r.to;
            });
            rep(i, first_edge[v], first_edge[v + 1]) {
                auto &e = edges[i];
                edges[e.rev].rev = i;
            }
        }

        copy(first_edge, first_edge + maxn, arc);
        fill(excess, excess + maxn, 0);
        excess[s] = INF, excess[t] = -INF;
        globalRelabel();

        rep(ie, first_edge[s], first_edge[s + 1])push(s, edges[ie]);

        for (; highest >= 0; highest--) {
            while (lst[highest] != -1) {
                int v = lst[highest];
                lst[highest] = nxt[v];
                if (height[v] == highest) {
                    discharge(v);
                    if (work > 4 * maxn)
                        globalRelabel();
                }
            }
        }
        return excess[t] + INF;
    }
};


int main() {
    ios::sync_with_stdio(false);
    int n, m; cin >> n >> m;
    int s, t; cin >> s >> t;
    s = s - 1, t = t - 1;
    vector<array<int, 3>> v(m);
    vector<int> degs(n);

    for (auto &x : v) {
        int a, b, c; cin >> a >> b >> c;
        x[0] = a - 1;
        x[1] = b - 1;
        x[2] = c;
        ++degs[x[0]];
        ++degs[x[1]];
    }

    HLPP<1205, 120005> hlpp(degs, s, t);

    for (auto &x : v)
        hlpp.addEdge(x[0], x[1], x[2]);

    cout << hlpp.calc() << '\n';

}
```

#### 费用流

```cpp
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
**预流推送费用流**
```cpp
constexpr const ll LL_INF = 0x3f3f3f3f3f3f3f3f;

template <const int MAXV, class flowUnit, class costUnit, const int SCALE = 8> struct PushRelabelMinCostMaxFlow {
    struct Edge {
        int to;
        flowUnit cap, resCap;
        costUnit cost;
        int rev;
        Edge(int to, flowUnit cap, costUnit cost, int rev) : to(to), cap(cap), resCap(cap), cost(cost), rev(rev) {}
    };
    int cnt[MAXV * 2], h[MAXV], stk[MAXV], top;
    flowUnit FLOW_EPS, maxFlow, ex[MAXV];
    costUnit COST_INF, COST_EPS, phi[MAXV], bnd, minCost, negCost;
    vector<int> hs[MAXV * 2];
    vector<Edge> adj[MAXV];
    typename vector<Edge>::iterator cur[MAXV];
    PushRelabelMinCostMaxFlow(flowUnit FLOW_EPS, costUnit COST_INF, costUnit COST_EPS) : FLOW_EPS(FLOW_EPS),
    COST_INF(COST_INF), COST_EPS(COST_EPS) {}
    void addEdge(int v, int w, flowUnit flow, costUnit cost) {
        if (v == w) {
            if (cost < 0)
                negCost += flow * cost;

            return;
        }

        adj[v].emplace_back(w, flow, cost, int(adj[w].size()));
        adj[w].emplace_back(v, 0, -cost, int(adj[v].size()) - 1);
    }
    void init(int V) {
        negCost = 0;

        for (int i = 0; i < V; i++)
            adj[i].clear();
    }
    flowUnit getMaxFlow(int V, int s, int t) {
        auto push = [&](int v, Edge & e, flowUnit df) {
            int w = e.to;

            if (abs(ex[w]) <= FLOW_EPS && df > FLOW_EPS)
                hs[h[w]].push_back(w);

            e.resCap -= df;
            adj[w][e.rev].resCap += df;
            ex[v] -= df;
            ex[w] += df;
        };

        if (s == t)
            return maxFlow = 0;

        fill(h, h + V, 0);
        h[s] = V;
        fill(ex, ex + V, 0);
        ex[t] = 1;
        fill(cnt, cnt + V * 2, 0);
        cnt[0] = V - 1;

        for (int v = 0; v < V; v++)
            cur[v] = adj[v].begin();

        for (int i = 0; i < V * 2; i++)
            hs[i].clear();

        for (auto &&e : adj[s])
            push(s, e, e.resCap);

        if (!hs[0].empty())
            for (int hi = 0; hi >= 0;) {
                int v = hs[hi].back();
                hs[hi].pop_back();

                while (ex[v] > FLOW_EPS) {
                    if (cur[v] == adj[v].end()) {
                        h[v] = INT_MAX;

                        for (auto e = adj[v].begin(); e != adj[v].end(); e++)
                            if (e->resCap > FLOW_EPS && h[v] > h[e->to] + 1) {
                                h[v] = h[e->to] + 1;
                                cur[v] = e;
                            }

                        cnt[h[v]]++;

                            if (--cnt[hi] == 0 && hi < V)
                                for (int i = 0; i < V; i++)
                                    if (hi < h[i] && h[i] < V) {
                                        cnt[h[i]]--;
                                        h[i] = V + 1;
                                    }

                            hi = h[v];
                    } else if (cur[v]->resCap > FLOW_EPS && h[v] == h[cur[v]->to] + 1)
                        push(v, *cur[v], min(ex[v], cur[v]->resCap));
                    else
                        cur[v]++;
                }

                while (hi >= 0 && hs[hi].empty())
                    hi--;
            }

        return maxFlow = -ex[s];
    }
    pair<flowUnit, costUnit> getMaxFlowMinCost(int V, int s = -1, int t = -1) {
        auto costP = [&](int v, const Edge & e) {
            return e.cost + phi[v] - phi[e.to];
        };
        auto push = [&](int v, Edge & e, flowUnit df, bool pushToStack) {
            if (e.resCap < df)
                df = e.resCap;

            int w = e.to;
            e.resCap -= df;
            adj[w][e.rev].resCap += df;
            ex[v] -= df;
            ex[w] += df;

            if (pushToStack && FLOW_EPS < ex[e.to] && ex[e.to] <= df + FLOW_EPS)
                stk[top++] = e.to;
        };
        auto relabel = [&](int v, costUnit delta) {
            phi[v] -= delta + bnd;
        };
        auto lookAhead = [&](int v) {
            if (abs(ex[v]) > FLOW_EPS)
                return false;

            costUnit delta = COST_INF;

            for (auto &&e : adj[v]) {
                if (e.resCap <= FLOW_EPS)
                    continue;

                costUnit c = costP(v, e);

                if (c < -COST_EPS)
                    return false;
                else
                    delta = min(delta, c);
            }

            relabel(v, delta);
            return true;
        };
        auto discharge = [&](int v) {
            costUnit delta = COST_INF;

            for (int i = 0; i < int(adj[v].size()); i++) {
                Edge &e = adj[v][i];

                if (e.resCap <= FLOW_EPS)
                    continue;

                if (costP(v, e) < -COST_EPS) {
                    if (lookAhead(e.to)) {
                        i--;
                        continue;
                    }

                    push(v, e, ex[v], true);

                    if (abs(ex[v]) <= FLOW_EPS)
                        return;
                } else
                    delta = min(delta, costP(v, e));
            }

            relabel(v, delta);
            stk[top++] = v;
        };
        minCost = 0;
        bnd = 0;
        costUnit mul = 2 << __lg(V);

        for (int v = 0; v < V; v++)
            for (auto &&e : adj[v]) {
                minCost += e.cost * e.resCap;
                e.cost *= mul;
                bnd = max(bnd, e.cost);
            }

        maxFlow = (s == -1 || t == -1) ? 0 : getMaxFlow(V, s, t);
            fill(phi, phi + V, 0);
            fill(ex, ex + V, 0);

            while (bnd > 1) {
                bnd = max(bnd / SCALE, costUnit(1));
                top = 0;

                for (int v = 0; v < V; v++)
                    for (auto &&e : adj[v])
                        if (costP(v, e) < -COST_EPS && e.resCap > FLOW_EPS)
                            push(v, e, e.resCap, false);

                        for (int v = 0; v < V; v++)
                            if (ex[v] > FLOW_EPS)
                                stk[top++] = v;

                            while (top > 0)
                                discharge(stk[--top]);
            }

            for (int v = 0; v < V; v++)
                for (auto &&e : adj[v]) {
                    e.cost /= mul;
                    minCost -= e.cost * e.resCap;
                }

            return make_pair(maxFlow, (minCost /= 2) += negCost);
    }
};
// 点数 类型 类型
PushRelabelMinCostMaxFlow<650, ll, ll> mcmf(0, LL_INF, 0);

ll sgn(ll x) { return x * x; }

ll val(ll x, ll y, ll z) {
    return sgn(x) + sgn(y) + sgn(z);
}

struct star {
    int x, y, z, v;
}p[305];
// 下标从0开始
int main() {
    int n; scanf("%d", &n);
    for (int i = 1; i <= n; ++i) {
        int x, y, z, v;
        scanf("%d%d%d%d", &x, &y, &z, &v);
        p[i] = {x, y, z, v};
    }
    int S = n + n + 1, T = n + n + 2;
    mcmf.init(T);

    for (int i = 1; i <= n; i++) {
        mcmf.addEdge(S - 1, i - 1, 1, 0);
        mcmf.addEdge(i + n - 1, T - 1, 1, 0);
    }

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            mcmf.addEdge(i - 1, j + n - 1, 1, val(p[j].x, p[j].y, p[j].z + 1ll * (i - 1) * p[j].v));
        }
    }

    printf("%lld\n", mcmf.getMaxFlowMinCost(T, --S, T - 1).second);
    return 0;
}
```

#### 无源汇上下界可行流

**有源汇的话加一条从T -> S [0, inf) 的边即可**

```cpp
void solve() {
    
    n = gn(), m = gn();
    S = 0, T = n + 1;

    for (int i = S; i <= T; ++i) head[i] = -1;
    
    // i -> j high[i] - low[i] 
    for (int i = 1; i <= m; ++i) {
        road[i] = {gn(), gn(), gn(), gn()};
        road[i].id = add_net_edge(road[i].x, road[i].y, road[i].high - road[i].low);
        in[road[i].y] += road[i].low;
        out[road[i].x] += road[i].low;
    }
    // S -> i in[i] > out[i] i -> T in[i] < out[i] 
    for (int i = 1; i <= n; ++i) {
        if (in[i] > out[i]) add_net_edge(S, i, in[i] - out[i]);
        else if (in[i] < out[i]) add_net_edge(i, T, out[i] - in[i]);
    }
    dinic();

    for (int i = head[S]; ~i; i = s[i].net) {
        if (s[i].w != 0) {
            puts("NO");
            return ;
        }
    }

    puts("YES");
    for (int i = 1; i <= m; ++i) {
        cout << road[i].low + s[road[i].id ^ 1].w << '\n';
    }
    cout << endl;
}
```

#### 有源汇上下界最大流、最小流

最小流最后一步改成减去汇点到源点的流就好

```cpp
struct node{
    int to, net;
    ll w;
}s[M * 2];
int dep[N], cur[N], tot = -1, head[N], S, T, n, m, ss, tt; // ss tt 真正的源汇点 S T 超源超汇

void add(int x, int y, int z) {
    s[++tot] = {y, head[x], z};
    head[x] = tot;
}

void add_net_edge (int x, int y, int z) {
    add(x, y, z);
    add(y, x, 0);
}

bool bfs(int st, int ed) {
    memset(dep, 0, sizeof dep);
    dep[st] = 1;
    queue<int> q;
    q.push(st);
    while (not q.empty()) {
        int now = q.front();
        q.pop();
        for (int i = head[now]; ~i; i = s[i].net) {
            int to = s[i].to;
            if (not dep[to] and s[i].w > 0) {
                dep[to] = dep[now] + 1;
                q.push(to);
            }
        }
    }
    return dep[ed];
}

ll dfs (int u, ll flo, int ed) {
    if (u == ed) return flo;
    ll del = 0;
    for (int i = cur[u]; (~i) and flo; i = s[i].net) {
        cur[u] = i;
        int to = s[i].to;
        if (dep[to] != dep[u] + 1 or s[i].w <= 0) continue;
        ll x = dfs(to, min(flo, s[i].w), ed);
        flo -= x; del += x;
        s[i].w -= x; s[i ^ 1].w += x;
    }
    if (not del) dep[u] = -2;
    return del;
}

int dinic(int st, int ed) {
    int ans = 0;
    while (bfs(st, ed)) {
        for (int i = S; i <= T; ++i) cur[i] = head[i];
        ans += dfs(st, (1 << 30), ed);
    }
    return ans;
}

int in[N], out[N];

int main() {
    n = gn(), m = gn(), ss = gn(), tt = gn();
    S = 0, T = n + 1;

    for (int i = S; i <= T; ++i) head[i] = -1;
    for (int i = 1; i <= m; ++i) {
        int x = gn(), y = gn(), low = gn(), high = gn();
        add_net_edge(x, y, high - low);
        in[y] += low;
        out[x] += low;
    }
    for (int i = 1; i <= n; ++i) {
        if (in[i] > out[i]) add_net_edge(S, i, in[i] - out[i]);
        else if (in[i] < out[i]) add_net_edge(i, T, out[i] - in[i]);
    }

    add_net_edge(tt, ss, INF);
    dinic(S, T);

    for (int i = head[S]; ~i; i = s[i].net) {
        if (s[i].w != 0) {
            puts("please go home to sleep");
            return 0;
        }
    }

    ll flow = s[tot].w;
    s[tot].w = s[tot - 1].w = 0;
    cout << flow + dinic(ss, tt) << endl;
}
```



### 树的直径

```cpp
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

### 最近公共祖先

**倍增求法**

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
```

**树链剖分求法**

```cpp
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



### 树的重心

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

### 树上路径交

A->B、C->D的路径交为
LCA(A, C), LCA(A, D), LCA(B, C), LCA(B, D)
其中深度最深的两点之间的路径

### 树链剖分

### 树上倍增
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

### 树上启发式合并

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



### 最小斯坦纳树

```cpp
/*
 *  Steiner Tree：求，使得指定K个点连通的生成树的最小总权值
 *  K 为特殊点的个数
 *  endSt=1<<K
 *  dp[i][state] 表示以i为根，连通状态为state的生成树值
 */

priority_queue<pair<int, int> > q;
int dp[(1 << K) + 2][N], endSt;

void initSteinerTree () {
    memset(dp, 0x3f, sizeof dp);
    endSt = 1 << k;
    for (int i = 1; i <= k; ++i) {
        dp[(1 << (i - 1))][i] = 0;
    }
}

void dijkstra(int state) {
    while (not q.empty()) {
        int now = q.top().second, dis = -q.top().first;
        q.pop();
        if (dp[state][now] < dis) continue;
        for (int i = head[now]; ~i; i = s[i].net) {
            int to = s[i].to;
            if (dp[state][to] > dp[state][now] + s[i].w) {
                dp[state][to] = dp[state][now] + s[i].w;
                q.push({-dp[state][to], to});
            }
        }
    }
}

int SteinerTree() {
    for (int state = 1; state < endSt; ++state) {
        for (int i = 1; i <= n; ++i) {
            for (int sub = state; sub; sub = (sub - 1) & state) {
                dp[state][i] = min(dp[state][i], dp[sub][i] + dp[state - sub][i]);
            }
            if (dp[state][i] != INF) q.push({-dp[state][i], i});
        }
        dijkstra(state);
    }

    int ans = INF;

    for (int i = 1; i <= n; ++i) {
        ans = min(ans, dp[(1 << k) - 1][i]);
    }

    return ans;
}
```

### 树上点分治

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