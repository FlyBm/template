# 博弈论

## 经典博弈

### Anti-Nim

> Nim博弈的前提下，取走最后一个石子的人为输。判断先手是否必胜
>
> 题目来源：HDU 1907

先手胜的局面当前仅当：

- 所有堆石子数都为1且有偶数堆（即SG值为0）
- 至少有一堆石子数大于1且SG值不为0

其余情况后手必胜

![img](https://img2018.cnblogs.com/i-beta/1417592/201911/1417592-20191127195332910-141546231.png)

### Anti-SG 和 SJ定理

- Anti-SG 游戏规定，决策集合为空的游戏者赢。

- Anti-SG 其他规则与 SG 游戏相同。

SJ定理为：对于任意一个 Anti-SG 游戏，如果我们规定当局面中所有的单一游戏的 SG 值为 0 时，游戏结束，则先手必胜当且仅当：

（1）游戏的 SG 函数不为 0 且游戏中某个单一游戏的 SG 函数大于1；

（2）游戏的 SG 函数为 0 且游戏中没有单一游戏的 SG 函数大于 1。

### Every-SG

- Every-SG 游戏规定，对于还没有结束的单一游戏，游戏者必须 对该游戏进行一步决策；
- Every-SG 游戏的其他规则与普通 SG 游戏相同

解法：在通过拓扑关系计算某一个状态点的 SG 函数时（事实上，我们只需 要计算该状态点的 SG 值是否为 0），对于 SG 值为 0 的点，我们需要知道 最快几步能将游戏带入终止状态，对于 SG 值不为 0 的点，我们需要知道 最慢几步游戏会被带入终止状态，我们用 step 函数来表示这个值。

![image-20201219224247874](C:\Users\22176\AppData\Roaming\Typora\typora-user-images\image-20201219224247874.png)

### 斐波那契博弈

> 1堆石子有n个,两人轮流取.先取者第1次可以取任意多个，但不能全部取完.以后每次取的石子数不能超过上次取子数的2倍。
>
> 题目来源：HDU 2516

当n为斐波那契数的时候后手必胜，否则先手必胜

### Nim-K

> n堆石子，每次从不超过k堆中取任意多个石子，最后不能取的人算失败。

结论： 把n堆石子的石子数用二进制表示，统计每一二进制位上的1的个数，若每一位上1的个数mod (k + 1)全为0，则必败。否则必胜

证明：<https://www.cnblogs.com/vongang/archive/2013/06/01/3112790.html>

### Anti-Nim-K

> n堆石子，每次从不超过k堆中取任意多个石子，取走最后石子的人算失败。

先手必胜:
1.石子规模都为1，且堆数mod (m + 1) != 1
2.石子规模不全为1，且当堆数以2进制表示时，存在某一二进制位上1的和mod(m + 1) != 0

证明同上

### New Nim

> 在第一个回合中，第一个游戏者可以直接拿走若干个整堆的火柴。可以一堆都不拿，但不可以全部拿走。第二回合也一样，第二个游戏者也有这样一次机会。从第三个回合（又轮到第一个游戏者）开始，规则和Nim游戏一样。如果你先拿，怎样才能保证获胜？如果可以获胜的话，还要让第一回合拿的火柴总数尽量小。

**结论**：为使后手必败，先手留给后手的必然是若干线性无关的数字，否则后手可以留下一个异或和为零的非空子集使得先手必败，故问题转化为拿走和最小的数字使得留下的数线性无关，即留下和最大的线性基，这样拿走的数量显然最少，找到和最大的线性基只需贪心的把数字从大到小加入到基中即可.

### 阶梯博弈

> 排成直线的格子上放有n个棋子，棋子$i$在左数第$p_i$个格子上。Geo and Bob轮流选择一个棋子向左移动。每次可以移动一格及其以上任意多格，但是不允许反超其他棋子，也不允许将两个棋子放在同一个格子内
>
> 题目来源：POJ 1704

题目原型： 有n堆石子，每堆石子的数量为$x_1, x_2, x_3, \dots, x_n$ ，A，B轮流操作，每次可以选第$k$堆中的任意多个石子放到第$k-1$堆中，第$1$堆中的石子可以放到第$0$堆中，最后无法操作的人为输。问A先手是否有必胜策略。

结论：用奇数层的石子堆做Nim博弈

感性思考：只观察奇数层石子的变化，我们可以发现如果操作当前$2k+1$层石子，那么其数量必然减少，同时转移的石子到$2k$层。同时如果$2k+1$层石子增加，自己也可以模仿对方操作从而使得对方这次操作无效化（对方把$2k+2$石子拿到$2k+1$，己方再把$2k+1$的石子拿到$2k$，如果只观察奇数层石子会发现没有任何变化。这样就相当于只用奇数层的石子在做Nim博弈

但是在本题中，我们需要一种等价转换的思想，我们可以把每两个棋子之间的格子看成一堆石子，右棋子左移格子减少可以看做拿石子，左棋子左移格子增加这个状态后手可以通过右棋子移动同样的步数使先手操作无效化。于是按照阶梯博弈处理即可，同样要注意的一个细节是棋子数量奇偶数会有影响。

### K倍动态减法

> 两人取一堆石子，石子有n个。 先手第一次不能全部取完但是至少取一个。之后每人取的个数不能超过另一个人上一次取的数的K倍。拿到最后一颗石子的赢。先手是否有必胜策略？若有，先手第一步最少取几个？ 来源：hdu2486

首先考虑k=1的情况，必败态为2的次方数，对必胜态，将个数二进制分解，每次取低位的1，对面肯定无法一次将剩下的所有取完。每次取最低位会必胜。

对于k=2的情况，必败态为fibonacci数，一个很重要的性质就是任意整数n都可以分解为不相邻的fibonacci数相加，不相邻的2个数之间差距2倍以上，因此可以分解后每次取最小位。

对于k>2的情况，参考上面，我们需要构造一个数列使得每个整数可以由数列中若干个数相加并且这些数倍数差距大于k，我们用a来存这些数，用b来表示前i-1个能构成的最大的数，那么a[i+1]=b[i]+1；然后再构造b[i+1]，由于b[i+1]要用到a[i+1]，并且不相邻，因此要找到a[j]*k<a[i]，b[i+1]=a[i+1]+b[j]。

查询时只需要不断减去最大的a[i]直到0，最后剩下的就是第一次取的值。

```cpp
const int maxn = 2000000;
int a[maxn], b[maxn];
int main() {
  int t, n, k;
  cin >> t;
  for (int cas = 1; cas <= t; cas++) {
    cin >> n >> k;
    printf("Case %d: \n", cas);
    if (n <= k + 1) {
      printf("lose\n");
      continue;
    }
    a[0] = b[0] = 1;
    int i = 0, j = 0;
    while (a[i] < n) {
      i++;
      a[i] = b[i - 1] + 1;
      while (a[j + 1] * k < a[i]) j++;
      if (a[j] * k < a[i])
        b[i] = b[j] + a[i];
      else
        b[i] = a[i];
    }
    if (a[i] == n)
      printf("lose\n");
    else {
      int ans = 0;
      while (n) {
        if (n >= a[i]) {
          n -= a[i];
          ans = a[i];
        }
        i--;
      }
      cout << ans << endl;
    }
  }

  ```

### Nim积

我们对于一些二维 Nim游戏（好像更高维也行），可以拆分成两维单独的 Nim 然后求 Nim 积。

定义为$x \otimes y = \mathrm{mex}\{(a \otimes b) \oplus (a \otimes y) \oplus (x \otimes b), 0 \le a < x, 0 \le b < y\}$

![image-20201220010604739](C:\Users\22176\AppData\Roaming\Typora\typora-user-images\image-20201220010604739.png)

代码

```cpp
#include <cstdio>
#include <iostream>
using namespace std;

int m[2][2] = {0, 0, 0, 1};
int Nim_Multi_Power(int x, int y) {
  if (x < 2) return m[x][y];
  int a = 0;
  for (;; a++)
    if (x >= (1 << (1 << a)) && x < (1 << (1 << (a + 1)))) break;
  int m = 1 << (1 << a);
  int p = x / m, s = y / m, t = y % m;
  int d1 = Nim_Multi_Power(p, s);
  int d2 = Nim_Multi_Power(p, t);
  return (m * (d1 ^ d2)) ^ Nim_Multi_Power(m / 2, d1);
}

int Nim_Multi(int x, int y) {
  if (x < y) return Nim_Multi(y, x);
  if (x < 2) return m[x][y];
  int a = 0;
  for (;; a++)
    if (x >= (1 << (1 << a)) && x < (1 << (1 << (a + 1)))) break;
  int m = 1 << (1 << a);
  int p = x / m, q = x % m, s = y / m, t = y % m;
  int c1 = Nim_Multi(p, s);
  int c2 = Nim_Multi(p, t) ^ Nim_Multi(q, s);
  int c3 = Nim_Multi(q, t);
  return (m * (c1 ^ c2)) ^ c3 ^ Nim_Multi_Power(m / 2, c1);
}

int main() {
  int t, n, x, y;
  scanf("%d", &t);
  while (t--) {
    scanf("%d", &n);
    int res = 0;
    while (n--) {
      scanf("%d%d", &x, &y);
      res ^= Nim_Multi(x, y);
    }
    if (res)
      printf("Have a try, lxhgww.\n");
    else
      printf("Don't waste your time.\n");
  }
  return 0;
}
```

## 找规律

### HDU 3032 SG函数找规律

> Nim博弈，不过增加了一个操作是每次可以把某一堆石子分成两堆。

我们可以把每堆石子看做一个SG函数，那么当前$SG[i]$的计算方式除了统计前i-1个SG[i]的值以外，还要计算i的所有拆分成两堆的SG函数异或。 但是n太大了，所以需要打个表，找出一个规律。

### NC 20893 图中删边 + 猜结论

> 一个n个点的无向完全图，每次操作都可以从当前点选择一条边移动到另一个点，并且这条边不能再走。不能再操作为输，问先手是否必胜。

n = 2时显然先手胜 n = 3时先手必胜 所以我们推一推n = 4时的情况发现先手也存在必胜策略，故猜测除了n = 1时先手皆为必胜。

## 经典博弈变式

### HDU 2897 Bash博弈变式

> 一堆石子有n个石头，一次最少取p个，最多取q个，两人轮流取直到堆为空。最后一次取硬币的人为输。先取者是否有必胜策略？

显然1 ~ p为先取者必败，p+1 ~ p + q为先取者必胜。  p + q 为循环节

### NC 17865 Bash博弈变式

> 一个会在t = 0时刻爆炸的炸弹，每个人每次可以将炸弹调快[a,b]秒，每次扔给对面消耗1秒。问先手是否必胜

和HDU2897思路类似

### CodeForces 1451D 对称博弈

> 两个人在坐标轴上轮流走。起始在$(0,0)$，每次只能向右走$k$格或者向上走$k$格。要求坐标满足$x^2+y^2 \le d^2$。求先手是否必胜。

假设能走到的最远位置为(zx,zy),只要判断(zx+x,y)是否在边界内即可。 如果不在则先手必败，因为此时后手通过模仿先手的动作，一定可以到达(zx,zy)从而使先手无路可走。

### NC Contest 9752C Bash博弈变式

> n个贝壳放成一堆，A先手B后手，A每次可以从中取[1,p]个，B每次可以取[1,q]个。问两人都最优策略，求谁能获胜

若p=q则显然bash博弈，若p!=q那么这不是一个公平组合游戏。 但是通过分析我们可以知道 if n <= p, then A win.

if n > p, then max(p,q) person win.

### NC 14619 Nim博弈变式

> Nim博弈，不过要求是如果先取者第一次拿第k堆的物品，是否必胜。

我们的主要目的是进行第一步取石子操作之后使得所有$x_i$异或为0.

设当前取的第k堆石头数量为$x_k$,我们先排除第k堆石头，设剩余石头的异或和为$H$,则我们的目标就是找到一个取完的状态$\acute{x_k}$使得$H \oplus \acute{x_k} = 0$,当$x_k >= \acute{x_k}$时一定可行，反之不行.

设所有石头个数异或和为$sum$,则$H = sum \oplus x_k$,故我们只需要判断 $sum \oplus x_k <= x_k$即可

### HDU 1850 Nim博弈变式

> NIm博弈，不过问的是先手想赢，第一步有多少种取法。

与上题类似，不过变化在于每个堆都要判断一下

### HDU 1730 Nim博弈变式

> 给出n行m列的棋盘，每行上有黑子和白子，A只能挪动黑子，B只能挪动白子，每次左右挪动可以任意格但不能越过对面棋子，要求A先手，是否有必胜策略。

显然可以分解成n个组合游戏，对于每个组合游戏我们可以把两个棋子之间的距离看做一个石子堆。那么这就是一个Nim博弈的等价转换。

### HDU 3533 三维Nim积

与二维Nim不同之处在于多了一个

```cpp
Nim_Multi(x, Nim_Multi(y, z));
```

### LightOj 1229

### LightOJ 1344

## SG函数

### NC 18388 Nim博弈变式 + SG函数

> NIm博弈，不过每次取每堆时，只能取“当前堆的石子个数的约数”的个数。求第一步有多少种不同的获胜策略.一个策略指的是，从哪堆石子中，取走多少颗石子。只要取的那一堆不同，或取的数目不同，都算不同的策略。

首先这个题目的SG函数计算就是要找到当前堆的所有可能状态。

其次这道题目和hdu1850不同之处在于**取同一堆的不同数量**仍记为不同策略. 所以最后判断的时候我们需要check的是$sum \oplus SG[ x[i] ] \oplus SG[x[i] - j] == 0 \wedge j| x[i]  $  

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

bool vis[100005];
int SG[100005], x[100005];

void getSG() {
  for (int i = 1; i <= 100000; i++) {
    memset(vis, 0, sizeof(vis));
    for (int j = 1; j * j <= i; j++) {
      if (i % j == 0) {
        vis[SG[i - j]] = true;
        vis[SG[i - i / j]] = true;
      }
    }
    for (int j = 0;; j++) {
      if (!vis[j]) {
        SG[i] = j;
        break;
      }
    }
  }
  // for(int i = 1; i <= 100; i++) printf("sg[%d]:%d\n", i, SG[i]);
}

int main() {
  getSG();
  int n;
  scanf("%d", &n);
  int sum = 0;
  for (int i = 1; i <= n; i++) {
    scanf("%d", &x[i]);
    sum ^= SG[x[i]];
  }
  if (sum == 0)
    printf("0");
  else {
    int ans = 0;
    for (int i = 1; i <= n; i++) {
      int now = sum ^ SG[x[i]];
      for (int j = 1; j * j <= x[i]; j++) {
        if (x[i] % j == 0) {
          if ((SG[x[i] - j] ^ now) == 0) ans++;
          if (j * j != x[i] && (SG[x[i] - x[i] / j] ^ now) == 0) ans++;
        }
      }
    }
    printf("%d", ans);
  }
  return 0;
}
```

### HDU1847 Bash博弈 + SG函数

> n个石子，每次可以抓取的数量是2的幂次，最后抓完牌的即是胜者。问先手是否必胜

计算从1~1000的SG值，SG为0说明先手必败。

### HDU2999 Nim函数变式 + SG函数

> 一堆石子有n个，每次能抓取的个数只能为连续的$a_1,a_2,\dots,a_n$个，求先手是否必胜

这个题目主要在于SG函数的状态转移，考虑当前石子个数为$x$时，我从第一个位置开始抓取$a_i$个石子，那么剩余的必然是$x - a_i$个石子，同时因为抓取的位置不同所以剩余的石子也不同，那么显然易见的是每次抓取后必然会产生两堆新石子。 因为产生的新石子可以看作Nim博弈，所以两者的异或和表示当前的一个可能转移状态。

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, m, k, a;
int SG[1005], num[111], pos;
set<int> S;

int dfs(int x) {
  if (SG[x] != -1) return SG[x];
  bool flag[1005];
  memset(flag, 0, sizeof(flag));
  for (int i = 1; i <= pos && num[i] <= x; i++) {
    for (int j = 1; j + num[i] - 1 <= x; j++) {
      SG[j - 1] = dfs(j - 1);
      SG[x - num[i] - j + 1] = dfs(x - num[i] - j + 1);
      flag[SG[j - 1] ^ SG[x - num[i] - j + 1]] = true;
    }
  }
  for (int i = 0; i <= 1000; i++) {
    if (!flag[i]) {
      return SG[x] = i;
    }
  }
}

int main() {
  while (scanf("%d", &n) != EOF) {
    S.clear();
    for (int i = 1; i <= 1000; i++) SG[i] = -1;
    SG[0] = 0;
    pos = 0;
    for (int i = 1; i <= n; i++) {
      scanf("%d", &a);
      S.insert(a);
    }
    for (auto now : S) num[++pos] = now;
    scanf("%d", &m);
    while (m--) {
      scanf("%d", &k);
      if (SG[k] == -1) SG[k] = dfs(k);
      if (SG[k] == 0)
        puts("2");
      else
        puts("1");
    }
  }
  return 0;
}
```

## 树上博弈

### Bamboo Stalks

> n条线段的bamboo stalks游戏是具有n条边的线形图。一步合法的操作是移除任意一条边，玩家轮流进行操作，最后一个进行操作的玩家获胜。n条线段的bamboo stalks游戏能够移动到任意更小线段数(0到n-1)的bamboo stalks游戏局面当中。

![img](https://img2018.cnblogs.com/i-beta/1417592/201911/1417592-20191127195625766-656456835.png)

可以看成nim博弈，例如，左边的三根竹竿构成的“森林”相当于具有石子数分别为3、4、5三堆石子的nim游戏。就我们所知，3^4^5=2，这是一个能够移动到P局面的N局面，办法是通过取走三根线段的竹竿上的第二根线段，留下一根。而结果变成右边的竹竿分布，而此时的SG值是0，是P局面。

### HDU 3094 Green Tree（单棵树删边）

> 给定一棵有根树，A和B分别轮流删边，删边后不与根联通的子树也一并删去。判断当前必胜或者必败！

思路：可以将树的每个结点的子树进行异或和，异或后的值生成的新树枝等价于原子树。（简单来说就是每个分叉点的分支异或和数值就是新树枝的长度）如下图所示

![img](https://img2018.cnblogs.com/i-beta/1417592/201911/1417592-20191127195700169-1276057524.png)

对于含环和多重根边的图这个结论同样适用！

```cpp
#include <bits/stdc++.h>
using namespace std;
int t;
int n;
vector<int> Edge[100005];

int dfs(int now, int fa) {
  int ans = 0;
  for (int i = 0; i < Edge[now].size(); i++) {
    int nxt = Edge[now][i];
    if (nxt != fa) {
      ans ^= dfs(nxt, now) + 1;
    }
  }
  return ans;
}

int main() {
  scanf("%d", &t);
  while (t--) {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) Edge[i].clear();
    while (n > 1) {
      n--;
      int x, y;
      scanf("%d%d", &x, &y);
      Edge[x].push_back(y);
      Edge[y].push_back(x);
    }
    // printf("%d\n", dfs(1, -1));
    if (dfs(1, -1) == 0)
      printf("Bob\n");
    else
      printf("Alice\n");
  }
  return 0;
}
```

### HDU 3197 多棵树删边

在上题的基础上，多棵树的删除可以类似于Nim游戏，求出多棵树的SG值的异或和即可。

### HDU 3590 多棵树删边 + Anti-Nim

在上题的基础上套用anti-nim的模型即可

### 树上有环的删边

 **The Fusion Principle**：任何环内的节点可以融合成一点而不会改变图的sg值。

-> 结论: 拥有**奇数条边的环**可简化为一条边，**偶数条边的环**可简化为一个节点。

![img](https://img2018.cnblogs.com/i-beta/1417592/201911/1417592-20191127195716328-1873340078.png)

可以变成

![fg7](C:\Users\22176\Desktop\fg7.jpeg)

### NC 19799 树链博弈

> 给定一棵 n 个点的树，其中 1 号结点是根，每个结点要么是黑色要么是白色
> 现在小 Bo 和小 Biao 要进行博弈，他们两轮流操作，每次选择一个黑色的结点将它变白，之后可以选择任意多个(可以不选)该点的祖先(不包含自己)，然后将这些点的颜色翻转，不能进行操作的人输
> 由于小 Bo 猜拳经常输给小 Biao，他想在这个游戏上扳回一城，现在他想问你给定了一个初始局面，是先手必胜还是后手必胜

思路1： 结论——若每层的黑子个数都为偶数则先手必败。

先看一条链的情况，发现只有在全为白时才会先手必败，再看二叉树时的情况画几个莽一发结论就能发现这个规律。

感性的证明：

若当前为先手必败态，那么先手无论是从哪层取石子，后手都可以在同层取石子 同时模仿先手的翻转动作使得其回到必败态。

思路2：

如果树上的某条分支是全白色那么显然可以看做删去了这条边。 于是我们可以把这个

### LightOJ 1355 边权大于1的删边

> 给一个树，每条边都有边权，每次操作可以减一，如果边权为0则删除这条边及其子树，不能操作则为输。

结论： 如果边权为1则就是green 博弈。 当边权大于1时，若边权为偶数则对SG值无贡献，若为奇数则SG[u] ^= SG[v] ^ 1

