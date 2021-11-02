# 计算几何

## 二维几何：点与向量

### 基本

```cpp
double eps = 1e-8;
int sgn(double k) {
  if (k > eps)
    return 1;
  else if (k < -eps)
    return -1;
  else
    return 0;
}
struct Point {
  double x, y;
  Point(double X = 0, double Y = 0) {
    x = X;
    y = Y;
  }
  Point operator+(const Point &a) { return Point(x + a.x, y + a.y); }
  Point operator-(const Point &a) { return Point(x - a.x, y - a.y); }
  Point operator*(const double &a) { return Point(x * a, y * a); }
  Point operator/(const double &a) { return Point(x / a, y / a); }
  double operator*(const Point &a) { return x * a.y - y * a.x; }
};
inline double Dot(Point a, Point b) { return a.x * b.x + a.y * b.y; }
inline double Cross(Point a, Point b) { return a.x * b.y - a.y * b.x; }
//另一种叉积公式
double xmult(double x1, double y1, double x2, double y2, double x0, double y0) {
  return (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
}
inline double Dis(Point a, Point b) {
  return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
```

### 点到线段距离

```cpp
double area_triangle(double x1, double y1, double x2, double y2, double x3,
                     double y3) {
  return fabs(xmult(x1, y1, x2, y2, x3, y3)) / 2;
}
double dis_ptoline(Point p, Point p1, Point p2) {
  double x1 = p1.x;
  double y1 = p1.y;
  double x2 = p2.x;
  double y2 = p2.y;
  double ex = p.x;
  double ey = p.y;
  double k, b, dis, tem1, tem2, t1, t2,
      yd = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
  t2 = sqrt((x2 - ex) * (x2 - ex) + (y2 - ey) * (y2 - ey));
  t1 = sqrt((x1 - ex) * (x1 - ex) + (y1 - ey) * (y1 - ey));
  dis = area_triangle(x1, y1, x2, y2, ex, ey) * 2 / yd;
  tem1 = sqrt(t1 * t1 - dis * dis);
  tem2 = sqrt(t2 * t2 - dis * dis);
  if (tem1 > yd || tem2 > yd) {
    if (t1 > t2) {
      //*px=x2;
      //*py=y2;
      return t2;
    } else {
      //*px=x1;
      //*py=y1;
      return t1;
    }
  }
  //*px=x1+(x2-x1)*tem1/yd;
  //*py=y1+(y2-y1)*tem1/yd;
  return dis;
}
```

### 判断点在多边形内部

```cpp
// 是否平行
bool parallel(const L& a, const L& b) { return !sgn(det(P(a), P(b))); }
// 是否在同一直线
bool l_eq(const L& a, const L& b) {
  return parallel(a, b) && parallel(L(a.s, b.t), L(b.s, a.t));
}
```

### 平面最近点对

```cpp
typedef struct Node {
  double x, y;
  int id;
} Node;
int n;
Node node[200100];
Node tran[200100];
double minx = 1e20;
bool same(double a, double b) {
  if (fabs(a - b) <= 1e-5) return true;
  return false;
}
int cmpx(Node a, Node b) {
  if (!same(a.x, b.x)) return a.x < b.x;
  return a.y < b.y;
}
int cmpy(Node a, Node b) {
  if (!same(a.y, b.y)) return a.y < b.y;
  return a.x < b.x;
}
void dist_minx(Node a, Node b) {
  double sum = sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
  minx = min(minx, sum);
}
void merge_node(int l, int mid, int r) {
  int t = 0;
  int i = l, j = mid + 1;
  while (i <= mid && j <= r) {
    if (node[i].y < node[j].y) {
      tran[++t] = node[i];
      i++;
      continue;
    } else {
      tran[++t] = node[j];
      j++;
      continue;
    }
  }
  while (i <= mid) tran[++t] = node[i++];
  while (j <= r) tran[++t] = node[j++];
  for (int i = l; i <= r; i++) node[i] = tran[i - l + 1];
}
void Blocking(int l, int r) {
  if (r - l <= 3) {
    for (int i = l; i < r; i++) {
      for (int j = i + 1; j < r; j++) {
        dist_minx(node[i], node[j]);
      }
    }
    sort(node + l, node + r + 1, cmpy);
    return;
  }
  int mid = (l + r) >> 1;
  double midx = node[mid].x;
  Blocking(l, mid);
  Blocking(mid + 1, r);
  merge_node(l, mid, r);
  vector<Node> Q;
  for (int i = l; i <= r; i++) {
    if (fabs(node[i].x - midx) >= minx) continue;
    for (int j = Q.size() - 1; j >= 0; j--) {
      if (fabs(Q[j].y - node[i].y) >= minx) break;
      dist_minx(Q[j], node[i]);
    }
    Q.push_back(node[i]);
  }
  Q.clear();
  return;
}
int main() {
  scanf("%d", &n);
  for (int i = 1; i <= n; i++) {
    scanf("%lf%lf", &node[i].x, &node[i].y);
    node[i].id = i;
  }
  sort(node + 1, node + n + 1, cmpx);
  Blocking(1, n);
  printf("%.4f\n", minx);
}
```

### 点基本公式

```c++
//点P绕点O旋转
Point rotate(Point p, Point o, double alpha) {
  p = p - o;
  o.x = o.x + p.x * cos(alpha) - p.y * sin(alpha);
  o.y = o.y + p.y * cos(alpha) + p.x * sin(alpha);
  return o;
}
//判断点是否在直线上
int judge_line(Point q, Point p1, Point p2) {
  if (sgn(Cross(q - p1, p2 - p1)) == 0) {
    return 1;
  } else
    return 0;
}
```

### 判断四个点是否能组成正方形

```c++
int judge_zheng(Point w[]) {
  int cnt = 0;
  for (int i = 1; i <= 4; i++)
    for (int j = i + 1; j <= 4; j++) len[++cnt] = Dis(w[i], w[j]);
  sort(len + 1, len + 1 + cnt);
  if (sgn(len[1] - len[2]) == 0 && sgn(len[3] - len[4]) == 0 &&
      sgn(len[1] - len[3]) == 0 && sgn(len[5] - len[6]) == 0 &&
      sgn(len[1] - len[5]) < 0) {
    return 1;
  } else {
    return 0;
  }
}
```



### 线与线

```cpp
//线段与线段相交
Point k;
int seg_inter_seg(Point a, Point b, Point c, Point d) {
  double s1, s2, s3, s4;
  int d1, d2, d3, d4;
  d1 = sgn(s1 = (b - a) * (c - a));
  d2 = sgn(s2 = (b - a) * (d - a));
  d3 = sgn(s3 = (d - c) * (a - c));
  d4 = sgn(s4 = (d - c) * (b - c));
  if ((d1 ^ d2) == -2 && (d3 ^ d4) == -2) {
    k = Point((c.x * s2 - d.x * s1) / (s2 - s1),
              (c.y * s2 - d.y * s1) / (s2 - s1));
    return 1;
  }
  if (d1 == 0 && sgn(Dot(a - c, b - c)) <= 0) {
    k = c;
    return 2;
  }
  if (d2 == 0 && sgn(Dot(a - d, b - d)) <= 0) {
    k = d;
    return 2;
  }
  if (d3 == 0 && sgn(Dot(c - a, d - a)) <= 0) {
    k = a;
    return 2;
  }
  if (d4 == 0 && sgn(Dot(c - b, d - b)) <= 0) {
    k = b;
    return 2;
  }
  // 0为不相交 1为严格相交 2为交交点
}
//直线与线段相交
int line_inter_seg(Point a, Point b, Point c, Point d)  // ab直线cd线段
{
  Point l1, l2;
  l1 = a;
  l2 = b;
  double step = 1e5;
  l2 = l2 - l1;
  l2 = l2 * step, l2 = l2 + l1;
  l1 = l1 - b;
  l1 = l1 * step, l1 = l1 + b;
  if (seg_inter_seg(l1, l2, st, ed))
    return 1;
  else
    return 0;
}
//直线与直线相交
Point intersection(Point u1, Point u2, Point v1, Point v2) {
  double t = ((u1 - v1) * (v1 - v2)) / ((u1 - u2) * (v1 - v2));
  return u1 + (u2 - u1) * t;
}
```

### 线段集相交（nlogn 玄学）

```C++
#include <bits/stdc++.h>
#define ll long long
using namespace std;
const int maxn = 1e5 + 50;
double eps = 1e-6;
int sgn(double k) {
  if (k > eps)
    return 1;
  else if (k < -eps)
    return -1;
  else
    return 0;
}
double ERROR = 0.0001;
struct line;
struct Point {
  double x;
  double y;
  struct line* belong1;
  struct line* belong2;
  int index;  // 0：上端点，1：下端点，2：交点
  bool operator()(Point* P1, Point* P2) { return P1->y < P2->y; }
  Point(double a = 0, double b = 0, line* c = nullptr, line* d = nullptr,
        int ind = 0)
      : x(a), y(b), belong1(c), belong2(d), index(ind){};
};
struct line {
  Point* first;
  Point* second;
};

struct ans {
  double x;
  double y;
  ans(double _x = 0, double _y = 0) {
    x = _x;
    y = _y;
  }
};

ans pans[maxn];

bool cmp(ans a, ans b) {
  if (sgn(a.y - b.y) == 0) {
    return sgn(a.x - b.x) < 0;
  }
  return sgn(a.y - b.y) < 0;
}

int ans_cnt = 0;

class Cutline {
 public:
  Cutline();
  ~Cutline();
  void GetRandomLine(int num);
  void FindIntersection();
  void HandleEvent(Point* event);
  Point* intersec(line* L1, line* L2);
  bool IsNewPoint(Point*);

 private:
  priority_queue<Point*, vector<Point*>, Point> P;
  map<double, line*> CurCutLine;
  vector<line*> Line;
  vector<Point*> intersectpoint;
  bool issamepoint(Point* P1, Point* P2);
};

Cutline::Cutline() {}

Cutline::~Cutline() {}

bool Cutline::issamepoint(Point* P1, Point* P2) {
  if (fabs(P1->x - P2->x) < 0.1 && fabs(P1->y - P2->y) < 0.1)
    return true;  // x坐标和y坐标同时都很相近
  return false;
}

bool Cutline::IsNewPoint(Point*) { return 1; }

Point* Cutline::intersec(line* L1, line* L2) {
  //计算交点坐标
  double x1 = L1->first->x;
  double y1 = L1->first->y;
  double x2 = L1->second->x;
  double y2 = L1->second->y;
  double x3 = L2->first->x;
  double y3 = L2->first->y;
  double x4 = L2->second->x;
  double y4 = L2->second->y;
  // cout << "(" << x1 << "," << y1 << ")  " << "(" << x2 << "," << y2 << ")  "
  // << "(" << x3 << "," << y3 << ")  " << "(" << x4 << "," << y4 << ")  " <<
  // endl;
  double k1 = (y1 - y2) / (x1 - x2);
  double k2 = (y3 - y4) / (x3 - x4);
  double x = (x1 * k1 - k2 * x3 + y3 - y1) / (k1 - k2);
  double y = k1 * (x - x1) + y1;
  if ((x - x1) * (x - x2) < 0 && (y - y1) * (y - y2) < 0 &&
      (x - x3) * (x - x4) < 0 && (y - y3) * (y - y4) < 0)  //判断是否在范围中
  {
    //构建新的交点
    // cout << "范围内" << endl;
    Point* temp;
    // temp1和temp2的归属，存在问题
    bool indx = L1->first->x < L2->first->x;
    if (indx)
      temp = new Point(x, y, L1, L2,
                       2);  // temp1是交点上方，在二叉树中靠左边的线段
    else
      temp = new Point(x, y, L2, L1, 2);
    return temp;
  }
  return nullptr;
}

void Cutline::GetRandomLine(int num) {
  double x1, y1, x2, y2;
  for (int i = 0; i < num; i++) {
    scanf("%lf %lf %lf %lf", &x1, &y1, &x2, &y2);
    line* temp = new line();
    Point* P1 = new Point(x1, y1);
    Point* P2 = new Point(x2, y2);
    P1->belong1 = temp;
    P2->belong1 = temp;
    if (P1->y > P2->y) {
      temp->first = P1;  // P1是上面的点
      temp->second = P2;
      P2->index = 1;
    } else {
      temp->first = P2;
      temp->second = P1;
      P1->index = 1;
    }
    Line.push_back(temp);
    /*file << count << " ";
    file << P1->x << " " << P1->y << " " << 0 << " ";//坐标
    file << 0 << " " << 0 << " " << 1 << " ";//法向量
    for (int j = 0;j < 10;j++) file << 0 << " ";
    file << endl;
    file << count << " ";
    file << P2->x << " " << P2->y << " " << 0 << " ";//坐标
    file << 0 << " " << 0 << " " << 1 << " ";//法向量
    for (int j = 0;j < 10;j++) file << 0 << " ";
    file << endl;*/
  }
}

void Cutline::FindIntersection() {
  for (auto x : Line) {
    P.push(x->first);
    P.push(x->second);
  }
  //初始化二分查找树
  while (!P.empty()) {
    auto temp = P.top();
    P.pop();
    HandleEvent(temp);
    // cout << CurCutLine.size() << endl;
  }
  /*ofstream file(outfile);
  int count = 1;
  for (auto x : intersectpoint)
  {
          file << count << " ";
          file << x->x << " " << x->y << " " << 0 << " ";//坐标
          file << 0 << " " << 0 << " " << 1 << " ";//法向量
          for (int j = 0;j < 10;j++) file << 0 << " ";
          file << endl;count++;
  }*/
}

void Cutline::HandleEvent(Point* event) {
  if (event->index == 2)  //交点
  {
    // cout << "交点" << endl;
    //去重
    if (!intersectpoint.empty() && issamepoint(event, intersectpoint.back()))
      return;  //重复点直接跳过
    intersectpoint.push_back(event);
    /*cout<<"test jiao = "<<event->x<<' '<<event->y<<endl;*/
    pans[++ans_cnt].x = event->x;
    pans[ans_cnt].y = event->y;
    auto jiaoxian1 = event->belong1;  //在交点上方靠左边的线段
    auto jiaoxian2 = event->belong2;
    auto it1 = CurCutLine.find(jiaoxian1->first->x);
    auto it2 = CurCutLine.find(jiaoxian2->first->x);
    if (it1 != CurCutLine.begin()) {
      auto it3 = --it1;
      it1++;
      auto jiaodian = intersec(it2->second, it3->second);
      if (jiaodian && jiaodian->y < event->y)  //
      {
        P.push(jiaodian);
      }
    }
    if (++it2 != CurCutLine.end()) {
      auto it4 = it2;
      it2--;
      auto jiaodian = intersec(it1->second, it4->second);
      if (jiaodian && jiaodian->y < event->y) P.push(jiaodian);
    } else
      it2--;
    //交换两条交线的位置
    //删除原来的两条线段
    CurCutLine.erase(it1);
    CurCutLine.erase(it2);
    //改变原来两条线段的上端点坐标，使之从交点开始
    jiaoxian1->first->y = event->y;
    jiaoxian2->first->y = event->y;
    jiaoxian1->first->x =
        event->x + ERROR;  //让之前在二叉树中靠左边的线段，靠右边
    jiaoxian2->first->x = event->x - ERROR;
    CurCutLine[jiaoxian1->first->x] = jiaoxian1;
    CurCutLine[jiaoxian2->first->x] = jiaoxian2;
  } else if (event->index == 0)  //上端点
  {
    /*cout << "上端点" <<event->x<< endl;
    for (auto x : CurCutLine) cout << x.first << " ";
    cout << endl;*/
    CurCutLine[event->x] = event->belong1;  //插入线段
    //找左邻居和右邻居,要确保找到的线段确实是邻居
    auto it = CurCutLine.find(event->x);
    decltype(it) left, right;
    if (++it == CurCutLine.end())  //判断是否到右边界
    {
      it--;
    } else {
      right = it;
      it--;
      // cout << "右邻居：" << right->second->first->x << endl;
      auto jiaodian = intersec(right->second, it->second);
      if (jiaodian) P.push(jiaodian);
    }
    if (it == CurCutLine.begin())  //左边界
    {
    } else {
      left = --it;
      it++;
      // cout << "左邻居：" << left->second->first->x << endl;
      auto jiaodian = intersec(left->second, it->second);
      if (jiaodian) P.push(jiaodian);
    }
  } else  //下端点
  {
    // cout << "下端点" << endl;
    //从查找树中删除该线段
    auto it = CurCutLine.find(event->belong1->first->x);
    decltype(it) left, right;
    if (++it == CurCutLine.end())  //判断是否到右边界
    {
      CurCutLine.erase(--it);
      return;
    } else {
      right = it;
      it--;
    }
    if (it == CurCutLine.begin())  //左边界
    {
      CurCutLine.erase(it);
      return;
    } else {
      left = --it;
      it++;
    }
    CurCutLine.erase(it);
    auto jiaodian = intersec(left->second, right->second);
    if (jiaodian && jiaodian->y < event->y)  //交点在扫描线下方
    {
      P.push(jiaodian);
    }
  }
}

int main() {
  Cutline C;
  int num;
  scanf("%d", &num);
  C.GetRandomLine(num);
  C.FindIntersection();
  for (int i = 1; i <= ans_cnt; i++) {
    printf("%.2f %.2f\n", pans[i].x, pans[i].y);
  }
  return 0;
}
```



## 凸包/多边形

### 凸包/多边形面积

```cpp
// graham
Point p[maxn], s[maxn];
int n;
int top;
bool operator<(Point a, Point b) {
  double t = Cross((a - p[1]), (b - p[1]));
  if (t == 0) return Dis(a, p[1]) < Dis(b, p[1]);
  return t < 0;
}

void graham() {
  int k = 1;
  for (int i = 2; i <= n; i++)
    if (p[k].y > p[i].y || (p[k].y == p[i].y && p[k].x > p[i].x)) k = i;
  swap(p[1], p[k]);
  sort(p + 2, p + n + 1);
  s[++top] = p[1];
  s[++top] = p[2];
  for (int i = 3; i <= n; i++) {
    while (top > 1 && Cross((p[i] - s[top - 1]), (s[top] - s[top - 1])) <= 0)
      top--;
    s[++top] = p[i];
  }
}
// Melkan 给定简单多边形 On
int cross(point a, point b) { return a.x * b.y - b.x * a.y; }
double dis(point a, point b) {
  return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
double Melkman(vector<paint> p) {
  int n = p.size();
  vector<paint> q(n * 2 + 10);
  int head, tail;
  head = tail = n;
  q[tail++] = p[0];

  int i;
  for (i = 0; i < n - 1; i++) {
    q[tail] = p[i];

    if (cross(p[i] - q[head], p[i + 1] - q[head])) break;
  }
  if (n == 1) return 0;
  if (n == 2) return dis(p[0], p[1]);
  if (n == 3) {
    return dis(p[0], p[1]) + dis(p[0], p[2]) + dis(p[1], p[2]);
  }
  q[--head] = q[++tail] = p[++i];
  if (cross(q[n + 1] - q[n], q[n + 2] - q[n]) < 0) swap(q[n], q[n + 1]);
  for (++i; i < n; i++) {
    if (cross(q[tail] - q[tail - 1], p[i] - q[tail - 1]) > 0 &&
        cross(q[head] - q[head + 1], p[i] - q[head + 1]) < 0)
      continue;
    while (tail - head > 1 &&
           cross(q[tail] - q[tail - 1], p[i] - q[tail - 1]) <= 0)
      tail--;
    q[++tail] = p[i];
    while (tail - head > 1 &&
           cross(q[head] - q[head + 1], p[i] - q[head + 1]) >= 0)
      head++;
    q[--head] = p[i];
  }
  double ans = 0;
  for (int i = head; i < tail; i++) {
    ans += dis(q[i], q[i + 1]);
  }
  return ans;
}
//计算多边形面积
inline double get_area(Point p[]) {
  double sum = 0;
  for (int i = 1; i <= n; i++) {
    if (i != n) {
      sum += Cross(p[i], p[i + 1]);
    } else {
      sum += Cross(p[i], p[1]);
    }
  }
  sum = sum / 2.0;
  return fabs(sum);
}
```

### 最大空凸包

```
inline double Sqr(double a) { return a * a; }
inline bool operator<(Point a, Point b) {
  return sgn(b.y - a.y) > 0 || sgn(b.y - a.y) == 0 && sgn(b.x - a.x) > 0;
}
inline double Max(double a, double b) { return a > b ? a : b; }
inline double Length(Point a) { return sqrt(Sqr(a.x) + Sqr(a.y)); }
Point dot[maxn], List[maxn];
double opt[maxn][maxn];
int seq[maxn], n, len;
double ans;
bool Compare(Point a, Point b) {
  int tmp = sgn(Cross(a, b));
  if (tmp != 0) return tmp > 0;
  tmp = sgn(Length(b) - Length(a));
  return tmp > 0;
}
void Solve(int vv) {
  int i, j, t, blen;
  for (i = len = 0; i < n; i++) {
    if (dot[vv] < dot[i]) List[len++] = dot[i] - dot[vv];
  }
  for (i = 0; i < len; i++) {
    for (j = 0; j < len; j++) opt[i][j] = 0;
  }
  sort(List, List + len, Compare);
  double v;
  for (t = 1; t < len; t++) {
    blen = 0;
    for (i = t - 1; i >= 0 && sgn(Cross(List[t], List[i])) == 0; i--)
      ;
    while (i >= 0) {
      v = Cross(List[i], List[t]) / 2;
      seq[blen++] = i;
      for (j = i - 1;
           j >= 0 && sgn(Cross(List[i] - List[t], List[j] - List[t])) > 0; j--)
        ;
      if (j >= 0) v += opt[i][j];
      ans = Max(ans, v);
      opt[t][i] = v;
      i = j;
    }
    for (i = blen - 2; i >= 0; i--) {
      opt[t][seq[i]] = Max(opt[t][seq[i]], opt[t][seq[i + 1]]);
    }
  }
}
int i;
double Empty() {
  ans = 0;
  for (i = 0; i < n; i++) Solve(i);
  return ans;
}
int main() {
  len = n;
  ans = Empty();
}
```



### 旋转卡壳

```cpp
// 求任意四点最大面积
inline double RC() {
  int a, b;
  double ans = 0;
  s[top + 1] = p[1];
  for (int i = 1; i <= n; i++) {
    a = i % top + 1;
    b = (i + 2) % top + 1;
    for (int j = i + 2; j <= top; j++) {
      while (a % top + 1 != j &&
             sgn(fabs(Cross(s[j] - s[i], s[a] - s[i])) -
                     fabs(Cross(s[j] - s[i], s[a + 1] - s[i])) <=
                 0)) {
        a = a % top + 1;
      }
      while (b % top + 1 != i &&
             sgn(fabs(Cross(s[b] - s[i], s[j] - s[i])) -
                     fabs(Cross(s[b + 1] - s[i], s[j] - s[i])) <=
                 0)) {
        b = b % top + 1;
      }
      ans = max(fabs(Cross(s[b] - s[i], s[j] - s[i])) +
                    fabs(Cross(s[j] - s[i], s[a] - s[i])),
                ans);
    }
  }
  return ans / 2.0;
}
//求多边形的长 // 平面最远点对
inline double RC() {
  double ans = 0;
  if (top == 2) {
    return Dis(s[1], s[2]);
  } else {
    s[++top] = s[1];
    int j = 3;
    for (int i = 1; i <= top; i++) {
      while (fabs(Cross(s[i] - s[i + 1], s[j] - s[i + 1])) <
             fabs(Cross(s[i] - s[i + 1], s[j + 1] - s[i + 1]))) {
        j = (j + 1) % top;
        if (j == 0) {
          j++;
        }
      }
      ans = max(ans, Dis(s[i], s[j]));
    }
  }
  return ans;
}
//求多边形的宽（使用独立凸包）
int n;
const ll INF = 0x3f3f3f3f;
Point p[maxn], s[maxn];
int top = 0;
void graham() {
  if (n == 1) {
    s[top++] = p[0];
  } else {
    for (int i = 0; i < n; i++) {
      while (top > 1 &&
             Cross(s[top - 2] - s[top - 1], s[top - 2] - p[i]) <= 0) {
        top--;
      }
      s[top++] = p[i];
    }
  }
}
double len(Point a, Point b, Point c) {
  double s = fabs(Cross(a - b, a - c));
  return s * 1.0 / Dis(a, b);
}

bool cmp(Point p1, Point p2) {
  long long tmp = Cross(p[0] - p1, p[0] - p2);
  if (tmp > 0) {
    return 1;
  } else if (tmp < 0) {
    return 0;
  } else {
    return Dis(p[0], p1) < Dis(p[0], p2);
  }
}
inline double RC() {
  double ans = INF;
  if (top == 2) {
    return Dis(s[1], s[2]);
  } else {
    int j = 1;
    for (int i = 1; i <= top; i++) {
      Point v = s[i - 1] - s[i % top];
      while (Cross(v, (s[(j + 1) % top] - s[j])) < 0) {
        j = (j + 1) % top;
      }
      ans = min(ans, len(s[i - 1], s[i % top], s[j]));
    }
    return ans;
  }
}
int main() {
  top = 0;
  Point p0(0, 1e9);
  scanf("%d", &n);
  int k = 0;
  for (int i = 0; i < n; i++) {
    scanf("%lf %lf", &p[i].x, &p[i].y);
    if (p0.y > p[i].y || (p0.y == p[i].y && p0.x > p[i].x)) {
      p0 = p[i];
      k = i;
    }
  }
  swap(p[0], p[k]);
  sort(p + 1, p + n, cmp);
  graham();
  double ans = RC();
  printf("%.10f\n", ans);
}
```

### 半平面交

```cpp
//求多凸边形面积交
struct Line {
  Point p1, p2;
  double ang;
};

double Cross(Point a, Point b) { return a.x * b.y - a.y * b.x; }

bool left(Line a, Point b) { return Cross(a.p2, b - a.p1) > eps; }

Point inter(Line a, Line b) {
  Point u = a.p1 - b.p1;
  double temp = Cross(b.p2, u) / Cross(a.p2, b.p2);
  return a.p1 + a.p2 * temp;
}

bool cmp(const Line &a, const Line &b) {
  return fabs(a.ang - b.ang) < eps ? left(a, b.p1) : a.ang < b.ang;
}
Point p[maxn], data_p[10][maxn];
Line a[maxn], q[maxn];
int poly_num[maxn];
double half_cal_area(int n) {
  int cnt = 0, m, tot = 1;
  for (int i = 1; i <= n; i++) {
    m = poly_num[i];
    for (int j = 1; j <= m; j++) {
      p[j] = data_p[i][j];
    }
    for (int j = 1; j <= m; j++) {
      a[++cnt].p1 = p[j], a[cnt].p2 = p[j] - p[j % m + 1];
      a[cnt].ang = atan2(a[cnt].p2.y, a[cnt].p2.x);
    }
  }
  sort(a + 1, a + 1 + cnt, cmp);
  for (int i = 2; i <= cnt; i++) {
    if (fabs(a[i].ang - a[i - 1].ang) > eps) {
      a[++tot] = a[i];
    }
  }
  q[1] = a[1];
  int l = 1, r = 1;
  for (int i = 2; i <= tot; i++) {
    while (l < r && left(a[i], p[r - 1])) r--;
    while (l < r && left(a[i], p[l])) l++;
    q[++r] = a[i];
    if (l < r) p[r - 1] = inter(q[r - 1], q[r]);
  }
  while (l < r && left(q[l], p[r - 1])) r--;
  p[r] = inter(q[l], q[r]), p[r + 1] = p[l];
  double ans = 0;
  for (int i = l; i <= r; i++) {
    ans += Cross(p[i], p[i + 1]);
  }
  return ans / 2.0;
}
//求能看到所有边的面积
const int maxn = 1505;
const double eps = 1e-8;
int n, pn, dq[maxn], top, bot;
struct Line {
  Point a, b;
  double angle;
  Line &operator=(Line l) {
    a.x = l.a.x;
    a.y = l.a.y;
    b.x = l.b.x;
    b.y = l.b.y;
    angle = l.angle;
    return *this;
  }
} l[maxn];

bool cmp(const Line &l1, const Line &l2) {
  int d = sgn(l1.angle - l2.angle);
  if (!d) {
    return sgn(Cross(l1.a, l2.a, l2.b)) < 0;
  }
  return d < 0;
}
void addLine(Line &l, double x1, double y1, double x2, double y2) {
  l.a.x = x1;
  l.a.y = y1;
  l.b.x = x2;
  l.b.y = y2;
  l.angle = atan2(y2 - y1, x2 - x1);
}
void getIntersect(Line l1, Line l2, Point &p) {
  double a1 = l1.b.y - l1.a.y;
  double b1 = l1.a.x - l1.b.x;
  double c1 = (l1.b.x - l1.a.x) * l1.a.y - (l1.b.y - l1.a.y) * l1.a.x;
  double a2 = l2.b.y - l2.a.y;
  double b2 = l2.a.x - l2.b.x;
  double c2 = (l2.b.x - l2.a.x) * l2.a.y - (l2.b.y - l2.a.y) * l2.a.x;
  p.x = (c2 * b1 - c1 * b2) / (a1 * b2 - a2 * b1);
  p.y = (c1 * a2 - c2 * a1) / (a1 * b2 - a2 * b1);
}
bool judge(Line l0, Line l1, Line l2) {
  Point p;
  getIntersect(l1, l2, p);
  return sgn(Cross(p, l0.a, l0.b)) > 0;
}
void HalfPlaneIntersect() {
  int i, j;
  sort(l, l + n, cmp);
  for (i = 0, j = 0; i < n; i++) {
    if (sgn(l[i].angle - l[j].angle) > 0) {
      l[++j] = l[i];
    }
  }
  n = j + 1;
  dq[0] = 0;
  dq[1] = 1;
  top = 1;
  bot = 0;
  //模拟双端队列 每一个进来的线淘汰前面的和后面的
  for (i = 2; i < n; i++) {
    while (top > bot && judge(l[i], l[dq[top]], l[dq[top - 1]])) {
      top--;
    }
    while (top > bot && judge(l[i], l[dq[bot]], l[dq[bot + 1]])) {
      bot++;
    }
    dq[++top] = i;
  }
  while (top > bot && judge(l[dq[bot]], l[dq[top]], l[dq[top - 1]])) {
    top--;
  }
  while (top > bot && judge(l[dq[top]], l[dq[bot]], l[dq[bot + 1]])) {
    bot++;
  }
  dq[++top] = dq[bot];
  for (pn = 0, i = bot; i < top; i++, pn++) {
    getIntersect(l[dq[i + 1]], l[dq[i]], p[pn]);
  }
}
double getArea() {
  if (pn < 3) {
    return 0;
  }
  double area = 0;
  for (int i = 1; i < pn - 1; i++) {
    area += Cross(p[0], p[i], p[i + 1]);
  }
  return fabs(area) / 2;
}
int main() {
  int t;
  scanf("%d", &t);
  while (t--) {
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
      scanf("%lf %lf", &p[i].x, &p[i].y);
    }
    for (int i = 0; i < n - 1; i++) {
      addLine(l[i], p[i].x, p[i].y, p[i + 1].x, p[i + 1].y);
    }
    addLine(l[n - 1], p[n - 1].x, p[n - 1].y, p[0].x, p[0].y);
    HalfPlaneIntersect();
    printf("%.2f\n", getArea());
  }
}
```

## 圆

```cpp
struct Circle {
  Point cir;
  double r;
} ；
```

### 最小圆覆盖

```cpp
struct Circle {
  Point cir;
  double r;
} ans;
Circle get_cir(Point a, double b) {
  Circle now;
  now.cir = a;
  now.r = b;
  return now;
}
Circle get_cir(Point a, Point b) {
  Circle now;
  now.cir = (a + b) / 2;
  now.r = Dis(a, b) / 2;
  return now;
}
Point rotate_nine(Point a) {
  Point fin;
  fin.x = a.y;
  fin.y = -a.x;
  return fin;
}
Circle get_cir(Point a, Point b, Point c) {
  Circle now;
  Point aa = a - b;
  Point bb = a - c;
  Point pov1 = (a + b) / 2;
  Point pov2 = (a + c) / 2;
  Point v1 = rotate_nine(aa);
  Point v2 = rotate_nine(bb);
  double t = Cross((pov2 - pov1), v2) / Cross(v1, v2);
  now.cir = pov1 + (v1 * t);
  now.r = Dis(now.cir, a);
  return now;
}
int main() {
  int n;
  scanf("%d", &n);
  if (n == 0) {
    return 0;
  }
  for (int i = 1; i <= n; i++) {
    scanf("%lf %lf", &p[i].x, &p[i].y);
  }
  random_shuffle(p + 1, p + n + 1);
  ans = get_cir(p[1], 0);
  for (int i = 2; i <= n; i++) {
    if (sgn(Dis(p[i], ans.cir) - ans.r) > 0) {
      ans = get_cir(p[i], 0);
      for (int j = 1; j <= i - 1; j++) {
        if (sgn(Dis(p[j], ans.cir) - ans.r) > 0) {
          ans = get_cir(p[i], p[j]);
          for (int k = 1; k <= j - 1; k++) {
            if (sgn(Dis(p[k], ans.cir) - ans.r) > 0) {
              ans = get_cir(p[i], p[j], p[k]);
            }
          }
        }
      }
    }
  }
  printf("%.2f %.2f ", ans.cir.x, ans.cir.y);
  printf("%.2f\n", ans.r);
}
```

### 圆的面积并（未测试）	

+ 圆和线的交点关于圆心是顺时针的

```cpp
#include <bits/stdc++.h>
#define R register
#define inline __inline__ __attribute__((always_inline))
#define fp(i, a, b) for (R int i = (a), I = (b) + 1; i < I; ++i)
#define fd(i, a, b) for (R int i = (a), I = (b)-1; i > I; --i)
#define go(u) for (int i = head[u], v = e[i].v; i; i = e[i].nx, v = e[i].v)
template <class T>
inline bool cmax(T &a, const T &b) {
  return a < b ? a = b, 1 : 0;
}
template <class T>
inline bool cmin(T &a, const T &b) {
  return a > b ? a = b, 1 : 0;
}
using namespace std;
const int N = 1005;
const double Pi = acos(-1.0);
struct Point {
  int x, y;
  inline Point() {}
  inline Point(R int xx, R int yy) : x(xx), y(yy) {}
  inline Point operator+(const Point &b) const {
    return Point(x + b.x, y + b.y);
  }
  inline Point operator-(const Point &b) const {
    return Point(x - b.x, y - b.y);
  }
  inline bool operator<(const Point &b) const {
    return x < b.x || (x == b.x && y < b.y);
  }
  inline bool operator==(const Point &b) const { return x == b.x && y == b.y; }
  inline double norm() { return sqrt(x * x + y * y); }
};
struct Cir {
  Point p;
  int r;
  inline bool operator<(const Cir &b) const {
    return p < b.p || p == b.p && r < b.r;
  }
  inline bool operator==(const Cir &b) const { return p == b.p && r == b.r; }
  inline double oint(R double t1, R double t2) {
    return r * (r * (t2 - t1) + p.x * (sin(t2) - sin(t1)) -
                p.y * (cos(t2) - cos(t1)));
  }
} c[N];
pair<double, int> st[N << 1];
int n;
double res;
double calc(int id) {
  int top = 0, cnt = 0;
  fp(i, 1, n) if (i != id) {
    double dis = (c[i].p - c[id].p).norm();
    if (c[id].r + dis <= c[i].r) return 0;
    if (c[i].r + dis <= c[id].r || c[i].r + c[id].r <= dis) continue;
    double del = acos((c[id].r * c[id].r + dis * dis - c[i].r * c[i].r) /
                      (2 * c[id].r * dis));
    double ang = atan2(c[i].p.y - c[id].p.y, c[i].p.x - c[id].p.x);
    double l = ang - del, r = ang + del;
    if (l < -Pi) l += 2 * Pi;
    if (r >= Pi) r -= 2 * Pi;
    if (l > r) ++cnt;
    st[++top] = make_pair(l, 1), st[++top] = make_pair(r, -1);
  }
  st[0] = make_pair(-Pi, 0), st[++top] = make_pair(Pi, 0);
  sort(st + 1, st + 1 + top);
  double res = 0;
  for (R int i = 1; i <= top; cnt += st[i++].second)
    if (!cnt) res += c[id].oint(st[i - 1].first, st[i].first);
  return res;
}
int main() {
  //  freopen("testdata.in","r",stdin);
  scanf("%d", &n);
  fp(i, 1, n) scanf("%d%d%d", &c[i].p.x, &c[i].p.y, &c[i].r);
  sort(c + 1, c + 1 + n), n = unique(c + 1, c + 1 + n) - c - 1;
  fp(i, 1, n) res += calc(i);
  printf("%.3f\n", res * 0.5);
  return 0;
}
```

### 圆与多边形面积并

```cpp
double det(Point a, Point b) { return a.x * b.y - a.y * b.x; }
double dot(Point a, Point b) { return a.x * b.x + a.y * b.y; }
Point operator*(Point a, double t) { return Point(a.x * t, a.y * t); }
Point operator+(Point a, Point b) { return Point(a.x + b.x, a.y + b.y); }
Point operator-(Point a, Point b) { return Point(a.x - b.x, a.y - b.y); }
double Length(Point a) { return sqrt(dot(a, a)); }

double Tri_cir_insection(Circle C, Point A, Point B) {
  Point oa = A - C.c, ob = B - C.c;
  Point ba = A - B, bc = C.c - B;
  Point ab = B - A, ac = C.c - A;
  double doa = Length(oa), dob = Length(ob), dab = Length(ab), r = C.r;
  double x =
      (dot(ba, bc) + sqrt(r * r * dab * dab - det(ba, bc) * det(ba, bc))) / dab;
  double y =
      (dot(ab, ac) + sqrt(r * r * dab * dab - det(ab, ac) * det(ab, ac))) / dab;
  double ts = det(oa, ob) * 0.5;

  if (sgn(det(oa, ob)) == 0) return 0;
  if (sgn(doa - C.r) < 0 && sgn(dob - C.r) < 0) {
    return det(oa, ob) * 0.5;
  } else if (dob < r && doa >= r)  // one in one out
  {
    return asin(ts * (1 - x / dab) * 2 / r / doa) * r * r * 0.5 + ts * x / dab;
  } else if (dob >= r && doa < r)  // one out one in
  {
    return asin(ts * (1 - y / dab) * 2 / r / dob) * r * r * 0.5 + ts * y / dab;
  } else if (fabs(det(oa, ob)) >= r * dab || dot(ab, ac) <= 0 ||
             dot(ba, bc) <= 0)  // 只有弧
  {
    if (dot(oa, ob) < 0) {
      if (det(oa, ob) < 0) {
        return (-acos(-1.0) - asin(det(oa, ob) / doa / dob)) * r * r * 0.5;
      } else {
        return (acos(-1.0) - asin(det(oa, ob) / doa / dob)) * r * r * 0.5;
      }
    } else {
      return asin(det(oa, ob) / doa / dob) * r * r * 0.5;
    }
  } else {
    return (asin(ts * (1 - x / dab) * 2 / r / doa) +
            asin(ts * (1 - y / dab) * 2 / r / dob)) *
               r * r * 0.5 +
           ts * ((x + y) / dab - 1);
  }
}
int main() {
  double r;
  int n;
  while (~scanf("%lf", &r)) {
    Circle C;
    C.c = Point(0, 0);
    C.r = r;
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) {
      scanf("%lf %lf", &p[i].x, &p[i].y);
    }
    p[n + 1] = p[1];
    double ans = 0;
    for (int i = 1; i <= n; i++) {
      ans += Tri_cir_insection(C, p[i], p[i + 1]);
    }
    printf("%.2f\n", fabs(ans));
  }
}
```

### 圆K次面积交

```cpp
#define sqr(x) ((x) * (x))
const int N = 1010;
double area[N];
int n;
struct cp {
  double x, y, r, angle;
  int d;
  cp() {}
  cp(double xx, double yy, double an = 0, int t = 0) {
    x = xx;
    y = yy;
    angle = an;
    d = t;
  }
  void get() {
    scanf("%lf %lf %lf", &x, &y, &r);
    d = 1;
  }
} cir[N], tp[N * 2];
int CirCrossCir(cp p1, double r1, cp p2, double r2, cp& cp1, cp& cp2) {
  double mx = p2.x - p1.x, sx = p2.x + p1.x, mx2 = mx * mx;
  double my = p2.y - p1.y, sy = p2.y + p1.y, my2 = my * my;
  double sq = mx2 + my2, d = -(sq - sqr(r1 - r2)) * (sq - sqr(r1 + r2));
  if (d + eps < 0) return 0;
  if (d < eps)
    d = 0;
  else
    d = sqrt(d);
  double x = mx * ((r1 + r2) * (r1 - r2) + mx * sx) + sx * my2;
  double y = my * ((r1 + r2) * (r1 - r2) + my * sy) + sy * mx2;
  double dx = mx * d, dy = my * d;
  sq *= 2;
  cp1.x = (x - dy) / sq;
  cp1.y = (y + dx) / sq;
  cp2.x = (x + dy) / sq;
  cp2.y = (y - dx) / sq;
  if (d > eps)
    return 2;
  else
    return 1;
}
bool circmp(const cp& u, const cp& v) { return sgn(u.r - v.r) < 0; }
bool cmp(const cp& u, const cp& v) {
  if (sgn(u.angle - v.angle)) return u.angle < v.angle;
  return u.d > v.d;
}
double calc(cp cir, cp cp1, cp cp2) {
  double ans = (cp2.angle - cp1.angle) * sqr(cir.r) - Cross(cir, cp1, cp2) +
               Cross(cp(0, 0), cp1, cp2);
  return ans / 2;
}
void CirUnion(cp cir[], int n) {
  cp cp1, cp2;
  sort(cir, cir + n, circmp);
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if (sgn(Dis(cir[i], cir[j]) + cir[i].r - cir[j].r) <= 0) {
        cir[i].d++;
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    int tn = 0, cnt = 0;
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      if (CirCrossCir(cir[i], cir[i].r, cir[j], cir[j].r, cp2, cp1) < 2)
        continue;
      cp1.angle = atan2(cp1.y - cir[i].y, cp1.x - cir[i].x);
      cp2.angle = atan2(cp2.y - cir[i].y, cp2.x - cir[i].x);
      cp1.d = 1;
      tp[tn++] = cp1;
      cp2.d = -1;
      tp[tn++] = cp2;
      if (sgn(cp1.angle - cp2.angle) > 0) {
        cnt++;
      }
    }
    tp[tn++] = cp(cir[i].x - cir[i].r, cir[i].y, pi, -cnt);
    tp[tn++] = cp(cir[i].x - cir[i].r, cir[i].y, -pi, cnt);
    sort(tp, tp + tn, cmp);
    int p;
    int s = cir[i].d + tp[0].d;
    for (int j = 1; j < tn; ++j) {
      p = s;
      s += tp[j].d;
      area[p] += calc(cir[i], tp[j - 1], tp[j]);
    }
  }
}
int main() {
  while (~scanf("%d", &n)) {
    for (int i = 0; i < n; i++) {
      cir[i].get();
    }
    memset(area, 0, sizeof(area));
    CirUnion(cir, n);
    for (int i = 1; i <= n; i++) {
      area[i] -= area[i + 1];
      printf("[%d] = %.3f\n", i, area[i]);
    }
  }
}
```

### 圆与圆交点（判断圆与圆相交）

```cpp
int intersection_circle_circle(Point c1, double r1, Point c2, double r2,
                               Point &p1, Point &p2) {
  double d = Dis(c1, c2);
  if (sgn(r1 + r2 - d) < 0) return 0;
  if (sgn(fabs(r1 - r2) - d) > 0) return 0;
  double a = atan2(c2.y - c1.y, c2.x - c1.x);
  double da = acos((r1 * r1 + d * d - r2 * r2) / (2 * r1 * d));
  p1 = Point(cos(a - da), sin(a - da)) * r1 + c1;
  p2 = Point(cos(a + da), sin(a + da)) * r1 + c1;
  if (p1 == p2) {
    return 1;
  }
  return 2;
}
```

## 二维几何公式

### 三角形

面积：
$$
S = absin(c) = \sqrt{p(p-a)(p-b)(p-c)}=Cross(A - B , A - C)
$$
中线长度：
$$
M_a（连接角A的中线）= \frac{1}{2}  \sqrt{2(b^2+c^2)-a^2}=\frac{1}{2}  \sqrt{(b^2+c^2)+2bccos(A)}
$$
角平分线：
$$
T_a（连接角A的角平分线）= \frac{2}{b+c} \sqrt{bcp(p - a)}
$$
高线：
$$
H_a（连接角A的角平分线）= bsin(C) = csin(B)
$$
内切圆半径：
$$
r=\frac{S}{p}=\frac{2 \pi r^2}{a+b+c}
$$
外切圆半径：
$$
R = \frac{abc}{4S} = \frac{a}{2sin(A)}
$$
三角形特殊三点共线：设X、Y、Z分别在△ABC的BC、CA、AB所在直线上，则X、Y、Z共线的充要条件是：
$$
\frac{AZ}{ZB} + \frac{BX}{XC} + \frac{CY}{YA}=1
$$
两直线一般式：
$$
a = y_1 - y_2 , b = x_2 - x_1 , c = x_1 y_2 - x_2 y_1
$$
欧拉公式三角形：

设△ABC的外心为O,内心为I,外接圆半径为R,内切圆半径为r,又记外心、内心的距离OI为d,则有
$$
d^2 = R^2 - 2  R  r
$$


### 四边形

D1 D2为对角线，M为对角线中点连线，A为对角线夹角

等式：
$$
a^2+b^2+c^2 = D_1^2 + D_2^2+4M^2
\\
S = \frac{D_1D_2sin(A)}{2}
$$
圆内接四边形独有：
$$
ac+bd=D_1D_2\\
S = \sqrt{(p-a)  (p-b)  (p-c)  (p-d)}
$$

### 正N边行

R为外接圆半径，r为内接圆半径，A为中心角

内角
$$
C = \frac{(n - 2)\pi}{n}
$$
边长：
$$
a = 2 \sqrt{(R^2-r^2)}=2Rsin(\frac{A}{2})=2rsin(\frac{A}{2})
$$
面积：
$$
S = \frac{nar}{2}
$$

### Pick定理

 一个计算点阵中顶点在格点上的多边形面积公式：
$$
S = \frac{a+b}{2} - 1
$$
其中a表示多边形内部的点数，b表示多边形边界上的点数，s表示多边形的面积。

设A(x1,y1),B(x2,y2),则三角形c边上的点的个数为gcd(x2-x1,y2-y1)+1

### 欧拉公式

空间：简单多面体的顶点数V、面数F及棱数E间有关系满足，平面V是图形P的顶点个数，F是图形P内的区域数，E是图形的边数
$$
V+F-E = 2
$$

### 圆

弧长
$$
l =rA
$$
弦长
$$
a = 2 \sqrt{(2hr - h^2)}=2rsin(\frac{A}{2})
$$
扇形面积
$$
S1=\frac{rl}{2}=\frac{r^2A}{2}
$$

### 椭圆

周长
$$
C=2\pi b+4(a-b)
$$
面积
$$
S=\pi ab
$$

### 棱柱

体积，S为底面积，h为高
$$
V=Sh
$$
侧面积，l为棱长，p为直截面周长
$$
S=lp
$$

### 棱锥

体积，S为底面积，h为高
$$
V=\frac{Sh}{3}
$$
（正棱锥）侧面积，l为斜高，p为底面周长
$$
S=\frac{lp}{2}
$$

### 棱台

体积，A1，A2为上下底面积，h为高
$$
V=\frac{(A_1+A_2+sqrt{(A_1A_2)})h}{3}
$$
(正棱台)侧面积，p1，p2为上下地面周长，h为高
$$
S=\frac{l(p_1+p_2)}{2}
$$

### 圆柱

侧面积
$$
S = 2\pi rh
$$
体积
$$
V = \pi r^2h
$$

### 圆锥

母线（任何圆锥的顶点到底面圆周上任意一点的线段叫做圆锥母线）
$$
l = \sqrt{h^2+r^2}
$$
侧面积
$$
S = \pi rl
$$
体积


$$
V = \frac{\pi r^2h}{3}
$$

### 圆台

母线
$$
l = \sqrt{h^2 - (r_1-r_2)^2}
$$
侧面积
$$
S = \pi(r_1 + r_2)l
$$
体积
$$
V=\frac{\pi (r_1^2+r_2^2+r_1r_2)h}{3}
$$

### 正多面体

正四面体：表面积$\sqrt{3}a^2$	体积$\frac{1}{12}\sqrt{2}a^3$	二面角角度$arccos(\frac{1}{3})$	外接球半径$\frac{\sqrt{6}}{4}a$	内切球半径$\frac{\sqrt{6}}{12}a$

立方体：表面积$6a^2$	体积$a^3$	二面角角度$90^。$	外接球半径$\sqrt{\frac{3}{4}}a$	内切球半径$\frac{1}{2}a$

正八面体：表面积$\sqrt{12}a^2$	体积$\frac{1}{3}\sqrt{2}a^3$	二面角角度$arccos(-\frac{1}{3})$	外接球半径$\frac{\sqrt{2}}{2}a$	内切球半径$\frac{\sqrt{6}}{6}a$

正十二面体：表面积$3\sqrt{25+10\sqrt{5}}a^2$	体积$\frac{1}{4}(15+7\sqrt{5})a^3$

### 圆环体

体积
$$
V=2 \pi^2r_1^2r_0
$$
表面积
$$
S = 4\pi^2r_0r_1
$$

## 三维

### 最小球覆盖

```c++
struct Point {
  double x, y, z;
  Point(double X = 0, double Y = 0, double Z = 0) {
    x = X;
    y = Y;
    z = Z;
  }
} p[maxn];
inline double Dis(Point a, Point b) {
  return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
              (a.z - b.z) * (a.z - b.z));
}
int n;
double Solve() {
  double Step = 200, ans = 1e9, mt;
  Point z = Point(0.0, 0.0, 0.0);
  int s = 0;
  while (Step > eps) {
    for (int i = 1; i <= n; ++i) {
      if (Dis(z, p[s]) < Dis(z, p[i])) s = i;
    }
    mt = Dis(z, p[s]);
    ans = min(ans, mt);
    z.x += (p[s].x - z.x) / mt * Step;
    z.y += (p[s].y - z.y) / mt * Step;
    z.z += (p[s].z - z.z) / mt * Step;
    Step *= 0.99;
  }
  return ans;
}
int main() {
  scanf("%d", &n);
  for (int i = 1; i <= n; i++) {
    scanf("%lf %lf %lf", &p[i].x, &p[i].y, &p[i].z);
  }
  printf("%.5f\n", Solve());
}
```

### 球与直线交点

```c++
vector<Point> ret;
void c_l_intersection(Point o, Point s, Point t, double r) {
  if (s == t) return;
  Point vec = (t - s) / Dis(t, s);
  Point onew = o - s;
  double dotov = Dot(onew, vec);
  double delta = 4 * (dotov * dotov - Dis(o, s) * Dis(o, s) + r * r);
  delta = sqrt(delta);
  if (sgn(delta) < 0) return;
  double t1 = dotov + delta / 2;
  ret.push_back(Point(s.x + t1 * vec.x, s.y + t1 * vec.y, s.z + t1 * vec.z));
  if (sgn(delta) > 0) {
    double t2 = t1 - delta;
    ret.push_back(Point(s.x + t2 * vec.x, s.y + t2 * vec.y, s.z + t2 * vec.z));
  }
}
```

### 三维凸包

```c++
using namespace std;
double eps = 1e-9;
struct TPoint {
  double x, y, z;
  TPoint() {}
  TPoint(double xx, double yy, double zz) : x(xx), y(yy), z(zz) {}
  TPoint operator-(const TPoint p) { return TPoint(x - p.x, y - p.y, z - p.z); }
  TPoint operator*(const TPoint p) {
    return TPoint(y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x);
  }
  double operator^(const TPoint p) { return x * p.x + y * p.y + z * p.z; }
};

TPoint dd;

struct fac  //判断是不是一个面
{
  int a, b, c;  //一个面三个点的编号
  bool ok;
};
TPoint xmult(TPoint u, TPoint v) {
  return TPoint(u.y * v.z - v.y * u.z, u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x);
}
double dmult(TPoint u, TPoint v) { return u.x * v.x + u.y * v.y + u.z * v.z; }
TPoint subt(TPoint u, TPoint v) {
  return TPoint(u.x - v.x, u.y - v.y, u.z - v.z);
}

double vlen(TPoint u) { return sqrt(u.x * u.x + u.y * u.y + u.z * u.z); }

TPoint pvec(TPoint a, TPoint b, TPoint c) {
  return xmult(subt(a, b), subt(b, c));
}

double Dis(TPoint a, TPoint b, TPoint c, TPoint d) {
  return fabs(dmult(pvec(a, b, c), subt(d, a))) / vlen(pvec(a, b, c));
}

struct T3dhull {
  int n;
  TPoint ply[N];
  int tri_cnt;
  fac tri[N];
  int vis[N][N];
  double dist(TPoint a) { return sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
  double area(TPoint a, TPoint b, TPoint c) { return dist((b - a) * (c - a)); }
  double volume(TPoint a, TPoint b, TPoint c, TPoint d) {
    return (b - a) * (c - a) ^ (d - a);
  }
  double ptoplane(TPoint &p, fac &f) {
    TPoint m = ply[f.b] - ply[f.a], n = ply[f.c] - ply[f.a], t = p - ply[f.a];
    return (m * n) ^ t;
  }
  void deal(int p, int a, int b) {
    int f = vis[a][b];
    fac add;
    if (tri[f].ok) {
      if ((ptoplane(ply[p], tri[f])) > eps) {
        dfs(p, f);
      } else {
        add.a = b;
        add.b = a;
        add.c = p;
        add.ok = 1;
        vis[p][b] = vis[a][p] = vis[b][a] = tri_cnt;
        tri[tri_cnt++] = add;
      }
    }
  }
  void dfs(int p, int cnt) {
    tri[cnt].ok = 0;
    deal(p, tri[cnt].b, tri[cnt].a);
    deal(p, tri[cnt].c, tri[cnt].b);
    deal(p, tri[cnt].a, tri[cnt].c);
  }
  bool same(int s, int e) {
    TPoint a = ply[tri[s].a], b = ply[tri[s].b], c = ply[tri[s].c];
    return (fabs(volume(a, b, c, ply[tri[e].a])) < eps &&
            fabs(volume(a, b, c, ply[tri[e].b])) < eps &&
            fabs(volume(a, b, c, ply[tri[e].c])) < eps);
  }
  void construct() {
    tri_cnt = 0;
    if (n < 4) return;
    bool tmp = true;
    for (int i = 1; i < n; i++) {
      if ((dist(ply[0] - ply[i])) > eps) {
        swap(ply[1], ply[i]);
        tmp = false;
        break;
      }
    }
    if (tmp) return;
    tmp = true;
    for (int i = 2; i < n; i++) {
      if ((dist((ply[0] - ply[1]) * (ply[1] - ply[i]))) > eps) {
        swap(ply[2], ply[i]);
        tmp = false;
        break;
      }
    }
    if (tmp) return;
    tmp = true;
    for (int i = 3; i < n; i++) {
      if (fabs((ply[0] - ply[1]) * (ply[1] - ply[2]) ^ (ply[0] - ply[i])) >
          eps) {
        swap(ply[3], ply[i]);
        tmp = false;
        break;
      }
    }
    if (tmp) return;
    fac add;
    for (int i = 0; i < 4; i++) {
      add.a = (i + 1) % 4, add.b = (i + 2) % 4, add.c = (i + 3) % 4, add.ok = 1;
      if ((ptoplane(ply[i], add)) > 0) {
        swap(add.b, add.c);
      }
      vis[add.a][add.b] = vis[add.b][add.c] = vis[add.c][add.a] = tri_cnt;
      tri[tri_cnt++] = add;
    }
    for (int i = 4; i < n; i++) {
      for (int j = 0; j < tri_cnt; j++) {
        if (tri[j].ok && ptoplane(ply[i], tri[j]) > eps) {
          dfs(i, j);
          break;
        }
      }
    }
    int cnt = tri_cnt;
    tri_cnt = 0;
    for (int i = 0; i < cnt; i++) {
      if (tri[i].ok) {
        tri[tri_cnt++] = tri[i];
      }
    }
  }
  double res() {
    double _min = 1e10;
    for (int i = 0; i < tri_cnt; i++) {
      double now = Dis(ply[tri[i].a], ply[tri[i].b], ply[tri[i].c], dd);
      if (_min > now) _min = now;
    }
    return _min;
  }
} hull;
int main() {
  int q;
  while (~scanf("%d", &hull.n)) {
    if (hull.n == 0) {
      break;
    }
    for (int i = 0; i < hull.n; i++) {
      scanf("%lf %lf %lf", &hull.ply[i].x, &hull.ply[i].y, &hull.ply[i].z);
    }
    hull.construct();
    scanf("%d", &q);
    for (int j = 0; j < q; j++) {
      scanf("%lf %lf %lf", &dd.x, &dd.y, &dd.z);
      printf("%.4f\n", hull.res());
    }
  }
}
```


