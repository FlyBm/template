### 动态规划
### 数学
### 计算几何
### 图论
#### 2-SAT
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
### 数据结构
### 字符串
