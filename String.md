## 字符串

[TOC]

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
int sa[N], rk[N], oldrk[N << 1], id[N], px[N], cnt[N], height[N];

bool cmp(int x, int y, int w) {
    return oldrk[x] == oldrk[y] and oldrk[x + w] == oldrk[y + w];
}

void SA(string s) {
    int i, m = 300, p, w, k;
    int n = s.length() - 1;
    for (i = 1; i <= n; ++i) ++cnt[rk[i] = s[i]];
    for (i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
    for (i = n; i >= 1; --i) sa[cnt[rk[i]]--] = i;

    for (w = 1; ; w <<= 1, m = p) {
        for (p = 0, i = n; i > n - w; --i) id[++p] = i;
        for (i = 1; i <= n; ++i) 
            if (sa[i] > w) id[++p] = sa[i] - w;
        memset(cnt, 0, sizeof cnt);
        for (i = 1; i <= n; ++i) ++cnt[px[i] = rk[id[i]]];
        for (i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
        for (i = n; i >= 1; --i) sa[cnt[px[i]]--] = id[i];
        memcpy(oldrk, rk, sizeof rk);
        for (p = 0, i = 1; i <= n; ++i) 
            rk[sa[i]] = cmp(sa[i], sa[i - 1], w) ? p : ++p;
        if (p == n) {
            for (i = 1; i <= n; ++i) sa[rk[i]] = i;
            break; 
        }
    }

    for (i = 1, k = 0; i <= n; ++i) {
        if (k) --k;
        while (s[i + k] == s[sa[rk[i] - 1] + k]) ++k;
        height[rk[i]] = k; 
    }
}
```

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
#define inducedSort(v) \
    fill_n(sa, n, -1); fill_n(cnt, m, 0); \                           
    for (int i = 0; i < n; i++) cnt[s[i]]++; \                          
    for (int i = 1; i < m; i++) cnt[i] += cnt[i-1]; \            
    for (int i = 0; i < m; i++) cur[i] = cnt[i]-1; \               
    for (int i = n1-1; ~i; i--) pushS(v[i]); \                  
    for (int i = 1; i < m; i++) cur[i] = cnt[i-1]; \             
    for (int i = 0; i < n; i++) if (sa[i] > 0 &&  t[sa[i]-1]) pushL(sa[i]-1); \
    for (int i = 0; i < m; i++) cur[i] = cnt[i]-1; \                    
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
