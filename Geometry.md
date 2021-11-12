[TOC]



# 计算几何

## 二维几何：点与向量

### 基本

```cpp
#define y1 yy1
#define nxt(i) ((i + 1) % s.size())
typedef double LD;
const LD PI = acos(-1);
const LD eps = 1E-10;
// ####
int sgn(LD x) { return fabs(x) < eps ? 0 : (x > 0 ? 1 : -1); }
// ####
struct L;
struct P;
// point == vector
typedef P V;

// point
struct P {
    LD x, y;
    explicit P(LD x = 0, LD y = 0): x(x), y(y) {}
    explicit P(const L& l);
};

// line
struct L {
    P s, t;
    L() {}
    L(P s, P t): s(s), t(t) {}
};

P operator + (const P& a, const P& b) { return P(a.x + b.x, a.y + b.y); }
P operator - (const P& a, const P& b) { return P(a.x - b.x, a.y - b.y); }
P operator * (const P& a, LD k) { return P(a.x * k, a.y * k); }
P operator / (const P& a, LD k) { return P(a.x / k, a.y / k); }

inline bool operator < (const P& a, const P& b) {
    return sgn(a.x - b.x) < 0 or
          (sgn(a.x - b.x) == 0 && sgn(a.y - b.y) < 0);
}
bool operator == (const P& a, const P& b) { 
    return !sgn(a.x - b.x) && !sgn(a.y - b.y); 
}

// line point vector 同源, line 可转换为P/V
P::P(const L& l) { *this = l.t - l.s; }

// IO
ostream &operator << (ostream &os, const P &p) {
    return (os << "(" << p.x << ", " << p.y << ") ");
}
istream &operator >> (istream &is, P &p) {
    return (is >> p.x >> p.y);
}

LD dist(const P& p) { return sqrt(p.x * p.x + p.y * p.y); }
LD dot(const V& a, const V& b) { return a.x * b.x + a.y * b.y; }
// 叉积 a x b
LD det(const V& a, const V& b) { return a.x * b.y - a.y * b.x; }
// 平行四边形面积  os × ot
LD cross(const P& s, const P& t, const P& o = P()) { return det(s - o, t - o); }

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


// --------------------------------------------
//+7惯用
double eps = 1e-8;

int sgn(double k) {
    if (k > eps) return 1;
    else if (k < -eps) return -1;
    else return 0;
}

struct Point {
    double x, y;

    Point(double X = 0, double Y = 0) {
        x = X;
        y = Y;
    }

    Point operator+(const Point &a) {
        return Point(x + a.x, y + a.y);
    }

    Point operator-(const Point &a) {
        return Point(x - a.x, y - a.y);
    }

    Point operator*(const double &a) {
        return Point(x * a, y * a);
    }

    Point operator/(const double &a) {
        return Point(x / a, y / a);
    }

    double operator*(const Point &a) {
        return x * a.y - y * a.x;
    }
};

inline double Dot(Point a, Point b) {
    return a.x * b.x + a.y * b.y;
}

inline double Cross(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}

//另一种叉积公式
double xmult(double x1, double y1, double x2, double y2, double x0, double y0) {
    return (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
}

inline double Dis(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
```

### 象限

```cpp
// 象限
int quad(P p) {
    int x = sgn(p.x), y = sgn(p.y);
    if (x > 0 && y >= 0) return 1;
    if (x <= 0 && y > 0) return 2;
    if (x < 0 && y <= 0) return 3;
    if (x >= 0 && y < 0) return 4;
    assert(0);
}

// 仅适用于参照点在所有点一侧的情况
struct cmp_angle {
    P p;
    bool operator () (const P& a, const P& b) {
//        int qa = quad(a - p), qb = quad(b - p);
//        if (qa != qb) return qa < qb;
        int d = sgn(cross(a, b, p));
        if (d) return d > 0;
        return dist(a - p) < dist(b - p);
    }
};
```

### 点、向量

```cpp
// 极角极坐标
LD arg(const P& a) { // arg in (-pi, pi]
    if (a.x > 0) 
        return atan(a.y / a.x);
    if (a.x == 0) {
        if (a.y > 0) return PI / 2;
        return -PI/2;
    } else { // a.x < 0
        if (a.y >= 0) return atan(a.y/ a.x) + PI;
        return atan(a.y / a.x) - PI;
    }
}
// 逆时针旋转 r 弧度
P rotation(const P& p, const LD& r) { return P(p.x * cos(r) - p.y * sin(r), p.x * sin(r) + p.y * cos(r)); }
// 逆时针
P RotateCCW90(const P& p) { return P(-p.y, p.x); }
// 顺时针
P RotateCW90(const P& p) { return P(p.y, -p.x); }
// 单位法向量
V normal(const V& v) { return V(-v.y, v.x) / dist(v); }
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
    vector <Node> Q;
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

### 线

```cpp
// 是否平行
bool parallel(const L& a, const L& b) {
    return !sgn(det(P(a), P(b)));
}
// 是否在同一直线
bool l_eq(const L& a, const L& b) {
    return parallel(a, b) && parallel(L(a.s, b.t), L(b.s, a.t));
}
```

### 点与线

```cpp
// 点在线段上  <= 0包含端点 < 0 则不包含
bool p_on_seg(const P& p, const L& seg) {
    P a = seg.s, b = seg.t;
    return !sgn(det(p - a, b - a)) && sgn(dot(p - a, p - b)) <= 0;
    // 点在线段上 <=> 平行 && a在两端点的中间 
}
//点在直线上
int judge_line(Point q , Point p1 , Point p2) {
    if ( sgn(Cross(q - p1 , p2 - p1)) == 0) return 1;
    else return 0;
}
// 点到直线距离
LD dist_to_line(const P& p, const L& l) {
    return fabs(cross(l.s, l.t, p)) / dist(l);
}
// 点到线段距离
LD dist_to_seg(const P& p, const L& l) {
    // l为单点
    if (l.s == l.t) return dist(p - l);
    V vs = p - l.s, vt = p - l.t;
    // p在 ts 方向
    if (sgn(dot(l, vs)) < 0) return dist(vs);
    // p在 st 方向
    else if (sgn(dot(l, vt)) > 0) return dist(vt);
    // p到直线距离即可
    else return dist_to_line(p, l);
}
// 点到线段距离2
double area_triangle(double x1, double y1, double x2, double y2, double x3, double y3) {
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

### 线与线

```cpp
// 求直线交 需要事先保证有界
P l_intersection(const L& a, const L& b) {
    LD s1 = det(P(a), b.s - a.s), s2 = det(P(a), b.t - a.s);
    return (b.s * s2 - b.t * s1) / (s2 - s1);
}
// 向量夹角的弧度
LD angle(const V& a, const V& b) {
    LD r = asin(fabs(det(a, b)) / dist(a) / dist(b));
    if (sgn(dot(a, b)) < 0) r = PI - r;
    return r;
}
// 线段和直线是否有交   1 = 规范，2 = 不规范
int s_l_cross(const L& seg, const L& line) {
    int d1 = sgn(cross(line.s, line.t, seg.s));
    int d2 = sgn(cross(line.s, line.t, seg.t));
    if ((d1 ^ d2) == -2) return 1; // proper
    if (d1 == 0 || d2 == 0) return 2;
    return 0;
}
// 线段的交   1 = 规范，2 = 不规范
int s_cross(const L& a, const L& b, P& p) {
    int d1 = sgn(cross(a.t, b.s, a.s)), d2 = sgn(cross(a.t, b.t, a.s));
    int d3 = sgn(cross(b.t, a.s, b.s)), d4 = sgn(cross(b.t, a.t, b.s));
    if ((d1 ^ d2) == -2 && (d3 ^ d4) == -2) { p = l_intersection(a, b); return 1; }
    if (!d1 && p_on_seg(b.s, a)) { p = b.s; return 2; }
    if (!d2 && p_on_seg(b.t, a)) { p = b.t; return 2; }
    if (!d3 && p_on_seg(a.s, b)) { p = a.s; return 2; }
    if (!d4 && p_on_seg(a.t, b)) { p = a.t; return 2; }
    return 0;
}
```

### 线段集相交（nlogn 玄学）

```cpp
#include <bits/stdc++.h>

#define ll long long
using namespace std;
const int maxn = 1e5 + 50;
double eps = 1e-6;

int sgn(double k) {
    if (k > eps) return 1;
    else if (k < -eps) return -1;
    else return 0;
}

double ERROR = 0.0001;
struct line;

struct Point {
    double x;
    double y;
    struct line *belong1;
    struct line *belong2;
    int index;//0：上端点，1：下端点，2：交点
    bool operator()(Point *P1, Point *P2) {
        return P1->y < P2->y;
    }

    Point(double a = 0, double b = 0, line *c = nullptr, line *d = nullptr, int ind = 0) : x(a), y(b), belong1(c),
                                                                                           belong2(d), index(ind) {};
};

struct line {
    Point *first;
    Point *second;
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

    void HandleEvent(Point *event);

    Point *intersec(line *L1, line *L2);

    bool IsNewPoint(Point *);

private:
    priority_queue<Point *, vector<Point *>, Point> P;
    map<double, line *> CurCutLine;
    vector<line *> Line;
    vector<Point *> intersectpoint;

    bool issamepoint(Point *P1, Point *P2);
};

Cutline::Cutline() { }

Cutline::~Cutline() { }

bool Cutline::issamepoint(Point *P1, Point *P2) {
    if (fabs(P1->x - P2->x) < 0.1 && fabs(P1->y - P2->y) < 0.1) return true;//x坐标和y坐标同时都很相近
    return false;
}

bool Cutline::IsNewPoint(Point *) {
    return 1;
}

Point *Cutline::intersec(line *L1, line *L2) {
    //计算交点坐标
    double x1 = L1->first->x;
    double y1 = L1->first->y;
    double x2 = L1->second->x;
    double y2 = L1->second->y;
    double x3 = L2->first->x;
    double y3 = L2->first->y;
    double x4 = L2->second->x;
    double y4 = L2->second->y;
    //cout << "(" << x1 << "," << y1 << ")  " << "(" << x2 << "," << y2 << ")  " << "(" << x3 << "," << y3 << ")  " << "(" << x4 << "," << y4 << ")  " << endl;
    double k1 = (y1 - y2) / (x1 - x2);
    double k2 = (y3 - y4) / (x3 - x4);
    double x = (x1 * k1 - k2 * x3 + y3 - y1) / (k1 - k2);
    double y = k1 * (x - x1) + y1;
    if ((x - x1) * (x - x2) < 0 && (y - y1) * (y - y2) < 0 && (x - x3) * (x - x4) < 0 &&
        (y - y3) * (y - y4) < 0) {//判断是否在范围中
        //构建新的交点
        //cout << "范围内" << endl;
        Point *temp;
        //temp1和temp2的归属，存在问题
        bool indx = L1->first->x < L2->first->x;
        if (indx) temp = new Point(x, y, L1, L2, 2);//temp1是交点上方，在二叉树中靠左边的线段
        else temp = new Point(x, y, L2, L1, 2);
        return temp;
    }
    return nullptr;
}

void Cutline::GetRandomLine(int num) {
    double x1, y1, x2, y2;
    for (int i = 0; i < num; i++) {
        scanf("%lf %lf %lf %lf", &x1, &y1, &x2, &y2);
        line *temp = new line();
        Point *P1 = new Point(x1, y1);
        Point *P2 = new Point(x2, y2);
        P1->belong1 = temp;
        P2->belong1 = temp;
        if (P1->y > P2->y) {
            temp->first = P1;//P1是上面的点
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
        //cout << CurCutLine.size() << endl;
    }
    /*ofstream file(outfile);
	int count = 1;
	for (auto x : intersectpoint) {
		file << count << " ";
		file << x->x << " " << x->y << " " << 0 << " ";//坐标
		file << 0 << " " << 0 << " " << 1 << " ";//法向量
		for (int j = 0;j < 10;j++) file << 0 << " ";
		file << endl;count++;
	}*/
}

void Cutline::HandleEvent(Point *event) {
    if (event->index == 2) {//交点
        //cout << "交点" << endl;
        //去重
        if (!intersectpoint.empty() && issamepoint(event, intersectpoint.back())) return;//重复点直接跳过
        intersectpoint.push_back(event);
        /*cout<<"test jiao = "<<event->x<<' '<<event->y<<endl;*/
        pans[++ans_cnt].x = event->x;
        pans[ans_cnt].y = event->y;
        auto jiaoxian1 = event->belong1;//在交点上方靠左边的线段
        auto jiaoxian2 = event->belong2;
        auto it1 = CurCutLine.find(jiaoxian1->first->x);
        auto it2 = CurCutLine.find(jiaoxian2->first->x);
        if (it1 != CurCutLine.begin()) {
            auto it3 = --it1;
            it1++;
            auto jiaodian = intersec(it2->second, it3->second);
            if (jiaodian && jiaodian->y < event->y) {
                P.push(jiaodian);
            }
        }
        if (++it2 != CurCutLine.end()) {
            auto it4 = it2;
            it2--;
            auto jiaodian = intersec(it1->second, it4->second);
            if (jiaodian && jiaodian->y < event->y) P.push(jiaodian);
        } else it2--;
        //交换两条交线的位置
        //删除原来的两条线段
        CurCutLine.erase(it1);
        CurCutLine.erase(it2);
        //改变原来两条线段的上端点坐标，使之从交点开始
        jiaoxian1->first->y = event->y;
        jiaoxian2->first->y = event->y;
        jiaoxian1->first->x = event->x + ERROR;//让之前在二叉树中靠左边的线段，靠右边
        jiaoxian2->first->x = event->x - ERROR;
        CurCutLine[jiaoxian1->first->x] = jiaoxian1;
        CurCutLine[jiaoxian2->first->x] = jiaoxian2;
    } else if (event->index == 0) {//上端点
        /*cout << "上端点" <<event->x<< endl;
		for (auto x : CurCutLine) cout << x.first << " ";
		cout << endl;*/
        CurCutLine[event->x] = event->belong1;//插入线段
        //找左邻居和右邻居,要确保找到的线段确实是邻居
        auto it = CurCutLine.find(event->x);
        decltype(it) left, right;
        if (++it == CurCutLine.end()) {//判断是否到右边界
            it--;
        } else {
            right = it;
            it--;
            //cout << "右邻居：" << right->second->first->x << endl;
            auto jiaodian = intersec(right->second, it->second);
            if (jiaodian) P.push(jiaodian);
        }
        if (it == CurCutLine.begin()) {//左边界
        } else {
            left = --it;
            it++;
            //cout << "左邻居：" << left->second->first->x << endl;
            auto jiaodian = intersec(left->second, it->second);
            if (jiaodian) P.push(jiaodian);
        }
    } else {//下端点
        //cout << "下端点" << endl;
        //从查找树中删除该线段
        auto it = CurCutLine.find(event->belong1->first->x);
        decltype(it) left, right;
        if (++it == CurCutLine.end()) {//判断是否到右边界{
            CurCutLine.erase(--it);
            return;
        } else {
            right = it;
            it--;
        }
        if (it == CurCutLine.begin()) {//左边界
            CurCutLine.erase(it);
            return;
        } else {
            left = --it;
            it++;
        }
        CurCutLine.erase(it);
        auto jiaodian = intersec(left->second, right->second);
        if (jiaodian && jiaodian->y < event->y) { //交点在扫描线下方
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



## 多边形

### 面积、凸包

```cpp
typedef vector<P> S;

// 点是否在多边形中 0 = 在外部 1 = 在内部 -1 = 在边界上
int inside(const S& s, const P& p) {
    int cnt = 0;
    FOR (i, 0, s.size()) {
        P a = s[i], b = s[nxt(i)];
        if (p_on_seg(p, L(a, b))) return -1;
        if (sgn(a.y - b.y) <= 0) swap(a, b);
        if (sgn(p.y - a.y) > 0) continue;
        if (sgn(p.y - b.y) <= 0) continue;
        cnt += sgn(cross(b, a, p)) > 0;
    }
    return bool(cnt & 1);
}

//另一种 点在多边形内 点1~n 边1~m
struct Line {
    Point p1, p2;
};

double xmulti(Point p1, Point p2, Point p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

double Max(double a, double b) {
    return a > b ? a : b;
}

double Min(double a, double b) {
    return a < b ? a : b;
}

bool ponls(Point q, Line l) {
    if (q.x > Max(l.p1.x, l.p2.x) || q.x < Min(l.p1.x, l.p2.x)
        || q.y > Max(l.p1.y, l.p2.y) || q.y < Min(l.p1.y, l.p2.y))
        return false;
    if (xmulti(l.p1, l.p2, q) == 0) return true;
    else return false;
}

bool judge(int pointnum, Point p[], Point q) {
    Line s;
    int c = 0;
    for (int i = 1; i <= pointnum; i++) {
        if (i == pointnum)
            s.p1 = p[pointnum], s.p2 = p[1];
        else
            s.p1 = p[i], s.p2 = p[i + 1];
        if (ponls(q, s))
            return true;
        if (s.p1.y != s.p2.y) {
            Point t;
            t.x = q.x - 1, t.y = q.y;
            if ((s.p1.y == q.y && s.p1.x <= q.x) || (s.p2.y == q.y && s.p2.x <= q.x)) {
                int tt;
                if (s.p1.y == q.y)
                    tt = 1;
                else if (s.p2.y == q.y)
                    tt = 2;
                int maxx;
                if (s.p1.y > s.p2.y)
                    maxx = 1;
                else
                    maxx = 2;
                if (tt == maxx)
                    c++;
            } else if (xmulti(s.p1, t, q) * xmulti(s.p2, t, q) <= 0) {
                Point lowp, higp;
                if (s.p1.y > s.p2.y)
                    lowp.x = s.p2.x, lowp.y = s.p2.y, higp.x = s.p1.x, higp.y = s.p1.y;
                else
                    lowp.x = s.p1.x, lowp.y = s.p1.y, higp.x = s.p2.x, higp.y = s.p2.y;
                if (xmulti(q, higp, lowp) >= 0)
                    c++;
            }
        }
    }
    if (c % 2 == 0) return false;
    else return true;
}

// 多边形面积，有向面积可能为负
LD polygon_area(const S& s) {
    LD ret = 0;
    FOR (i, 1, (LL)s.size() - 1)
        ret += cross(s[i], s[i + 1], s[0]);
    return ret / 2;
}

// 构建凸包 点不可以重复 < 0 边上可以有点， <= 0 则不能
// 会改变输入点的顺序
const int MAX_N = 1000;
S convex_hull(S& s) {
//    assert(s.size() >= 3);
    sort(s.begin(), s.end());
    S ret(MAX_N * 2);
    int sz = 0;
    FOR (i, 0, s.size()) {
        while (sz > 1 && sgn(cross(ret[sz - 1], s[i], ret[sz - 2])) <= 0) --sz;
        ret[sz++] = s[i];
    }
    int k = sz;
    FORD (i, (LL)s.size() - 2, -1) {
        while (sz > k && sgn(cross(ret[sz - 1], s[i], ret[sz - 2])) <= 0) --sz;
        ret[sz++] = s[i];
    }
    ret.resize(sz - (s.size() > 1));
    return ret;
}

P ComputeCentroid(const S &s) {
    P c(0, 0);
    LD scale = 6.0 * polygon_area(p);
    for (unsigned i = 0; i < s.size(); i++) {
        unsigned j = (i + 1) % s.size();
        c = c + (s[i] + s[j]) * (s[i].x * s[j].y - s[j].x * s[i].y);
    }
    return c / scale;
}
//+7凸包
Point p[maxn],s[maxn];
int n;
int top;
bool operator < (Point a,Point b) {
    double t=Cross((a-p[1]),(b-p[1]));
    if(t==0)return Dis(a,p[1])<Dis(b,p[1]);
    return t<0;
}

void graham() {
	int k = 1;
	for(int i = 2 ; i <= n ; i ++)
		if(p[k].y > p[i].y || (p[k].y == p[i].y && p[k].x > p[i].x) )
			k=i;
	swap(p[1] , p[k]);
	sort(p + 2 , p + n + 1);
	s[++ top] = p[1]; s[++ top] = p[2];
	for(int i = 3 ; i <= n; i ++) {
		while(top > 1 && Cross( (p[i] - s[top - 1]) , (s[top] - s[top - 1])) <= 0)
			top --;
		s[++ top] = p[i];
	}
}
//Melkman 简单多边形凸包 On
int cross(point a,point b) {
    return a.x*b.y-b.x*a.y;
}
double dis(point a,point b) {
    return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));
}
double Melkman(vector<paint>p) {
    int n=p.size();
    vector<paint>q(n*2+10);
    int head,tail;
     head=tail=n;
    q[tail++]=p[0];

    int i;
    for(i=0;i<n-1;i++) {
        q[tail]=p[i];

        if(cross(p[i]-q[head],p[i+1]-q[head])) break;
    }
    if(n==1) return 0;
    if(n==2) return dis(p[0],p[1]);
    if(n==3) {
        return dis(p[0],p[1])+dis(p[0],p[2])+dis(p[1],p[2]);
    }
    q[--head]=q[++tail]=p[++i];
    if(cross(q[n+1]-q[n],q[n+2]-q[n])<0) swap(q[n],q[n+1]);
    for(++i;i<n;i++) {
       if(cross(q[tail]-q[tail-1],p[i]-q[tail-1])>0
        &&cross(q[head]-q[head+1],p[i]-q[head+1])<0) continue;
      while(tail-head>1&&cross(q[tail]-q[tail-1],p[i]-q[tail-1])<=0)  tail--;
      q[++tail]=p[i];
      while(tail-head>1&&cross(q[head]-q[head+1],p[i]-q[head+1])>=0)  head++;
      q[--head]=p[i];

    }
    double ans=0;
    for(int i=head;i<tail;i++) {
        ans+=dis(q[i],q[i+1]);
    }
    return ans;
}
//最大空凸包
inline double Sqr(double a) {
	return a * a;
}
inline bool operator < (Point a,Point b) {
	return sgn(b.y - a.y) > 0 || sgn(b.y - a.y) == 0 && sgn(b.x - a.x) > 0;
}
inline double Max(double a,double b) {
	return a > b ? a : b;
}
inline double Length(Point a) {
	return sqrt(Sqr(a.x) + Sqr(a.y));
}
Point dot[maxn],List[maxn]; double opt[maxn][maxn];
int seq[maxn], n,len;   double ans;
bool Compare(Point a,Point b) {
	int tmp = sgn(Cross(a,b));
	if(tmp != 0)
		return tmp > 0;
	tmp = sgn(Length(b) - Length(a));
	return tmp > 0;
}
void Solve(int vv) {
	int i,j,t,blen;
	for(i = len = 0; i < n; i++) {
		if(dot[vv] < dot[i])
			List[len++] = dot[i] - dot[vv];
    }
	for(i = 0; i < len; i++) {
		for(j = 0; j < len; j++)
			opt[i][j] = 0;
	}
	sort(List,List + len,Compare);
	double v;
	for(t = 1; t < len; t++) {
		blen = 0;
		for(i = t - 1; i >= 0 && sgn(Cross(List[t],List[i])) == 0; i--);
		while(i >= 0) {
			v = Cross(List[i],List[t]) / 2;
			seq[blen++] = i;
			for(j = i - 1; j >= 0 && sgn(Cross(List[i] - List[t],List[j] - List[t])) > 0; j--);
			if(j >= 0) v += opt[i][j];
			ans = Max(ans,v);
			opt[t][i] = v;
			i = j;
		}
		for(i = blen - 2; i >= 0; i--) {
			opt[t][seq[i]] = Max(opt[t][seq[i]],opt[t][seq[i + 1]]);
		}
	}
}
int i;
double Empty() {
	ans = 0;
	for(i = 0; i < n; i++) Solve(i);
	return ans;
}
int main() {
	len = n;
	ans = Empty();
}
```

### 旋转卡壳

```cpp
LD rotatingCalipers(vector<P>& qs) {
    int n = qs.size();
    if (n == 2)
        return dist(qs[0] - qs[1]);
    int i = 0, j = 0;
    FOR (k, 0, n) {
        if (!(qs[i] < qs[k])) i = k;
        if (qs[j] < qs[k]) j = k;
    }
    LD res = 0;
    int si = i, sj = j;
    while (i != sj || j != si) {
        res = max(res, dist(qs[i] - qs[j]));
        if (sgn(cross(qs[(i+1)%n] - qs[i], qs[(j+1)%n] - qs[j])) < 0)
            i = (i + 1) % n;
        else j = (j + 1) % n;
    } 
    return res;
}

int main() {
    int n;
    while (cin >> n) {
        S v(n);
        FOR (i, 0, n) cin >> v[i].x >> v[i].y;
        convex_hull(v);
        printf("%.0f\n", rotatingCalipers(v));
    }
}

//+7
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
                   sgn(fabs(Cross(s[j] - s[i], s[a] - s[i])) - fabs(Cross(s[j] - s[i], s[a + 1] - s[i])) <= 0)) {
                a = a % top + 1;
            }
            while (b % top + 1 != i &&
                   sgn(fabs(Cross(s[b] - s[i], s[j] - s[i])) - fabs(Cross(s[b + 1] - s[i], s[j] - s[i])) <= 0)) {
                b = b % top + 1;
            }
            ans = max(fabs(Cross(s[b] - s[i], s[j] - s[i])) + fabs(Cross(s[j] - s[i], s[a] - s[i])), ans);
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
            while (fabs(Cross(s[i] - s[i + 1], s[j] - s[i + 1])) < fabs(Cross(s[i] - s[i + 1], s[j + 1] - s[i + 1]))) {
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
            while (top > 1 && Cross(s[top - 2] - s[top - 1], s[top - 2] - p[i]) <= 0) {
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
struct LV {
    P p, v; LD ang;
    LV() {}
    LV(P s, P t): p(s), v(t - s) { ang = atan2(v.y, v.x); }
};  // 另一种向量表示

bool operator < (const LV &a, const LV& b) { return a.ang < b.ang; }
bool on_left(const LV& l, const P& p) { return sgn(cross(l.v, p - l.p)) >= 0; }
P l_intersection(const LV& a, const LV& b) {
    P u = a.p - b.p; LD t = cross(b.v, u) / cross(a.v, b.v);
    return a.p + a.v * t;
}

S half_plane_intersection(vector<LV>& L) {
    int n = L.size(), fi, la;
    sort(L.begin(), L.end());
    vector<P> p(n); vector<LV> q(n);
    q[fi = la = 0] = L[0];
    FOR (i, 1, n) {
        while (fi < la && !on_left(L[i], p[la - 1])) la--;
        while (fi < la && !on_left(L[i], p[fi])) fi++;
        q[++la] = L[i];
        if (sgn(cross(q[la].v, q[la - 1].v)) == 0) {
            la--;
            if (on_left(q[la], L[i].p)) q[la] = L[i];
        }
        if (fi < la) p[la - 1] = l_intersection(q[la - 1], q[la]);
    }
    while (fi < la && !on_left(q[fi], p[la - 1])) la--;
    if (la - fi <= 1) return vector<P>();
    p[la] = l_intersection(q[la], q[fi]);
    return vector<P>(p.begin() + fi, p.begin() + la + 1);
}

S convex_intersection(const vector<P> &v1, const vector<P> &v2) {
    vector<LV> h; int n = v1.size(), m = v2.size();
    FOR (i, 0, n) h.push_back(LV(v1[i], v1[(i + 1) % n]));
    FOR (i, 0, m) h.push_back(LV(v2[i], v2[(i + 1) % m]));
    return half_plane_intersection(h);
}

//求多凸边形面积交
struct Line {
    Point p1, p2;
    double ang;
};

double Cross(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}

bool left(Line a, Point b) {
    return Cross(a.p2, b - a.p1) > eps;
}

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
struct C {
    P p; LD r;
    C(LD x = 0, LD y = 0, LD r = 0): p(x, y), r(r) {}
    C(P p, LD r): p(p), r(r) {}
};
```

### 三点求圆心

```cpp
P compute_circle_center(P a, P b, P c) {
    b = (a + b) / 2;
    c = (a + c) / 2;
    return l_intersection({b, b + RotateCW90(a - b)}, {c , c + RotateCW90(a - c)});
}
```

### 圆线交点、圆圆交点

+ 圆和线的交点关于圆心是顺时针的

```cpp
vector<P> c_l_intersection(const L& l, const C& c) {
    vector<P> ret;
    P b(l), a = l.s - c.p;
    LD x = dot(b, b), y = dot(a, b), z = dot(a, a) - c.r * c.r;
    LD D = y * y - x * z;
    if (sgn(D) < 0) return ret;
    ret.push_back(c.p + a + b * (-y + sqrt(D + eps)) / x);
    if (sgn(D) > 0) ret.push_back(c.p + a + b * (-y - sqrt(D)) / x);
    return ret;
}

vector<P> c_c_intersection(C a, C b) {
    vector<P> ret;
    LD d = dist(a.p - b.p);
    if (sgn(d) == 0 || sgn(d - (a.r + b.r)) > 0 || sgn(d + min(a.r, b.r) - max(a.r, b.r)) < 0)
        return ret;
    LD x = (d * d - b.r * b.r + a.r * a.r) / (2 * d);
    LD y = sqrt(a.r * a.r - x * x);
    P v = (b.p - a.p) / d;
    ret.push_back(a.p + v * x + RotateCCW90(v) * y);
    if (sgn(y) > 0) ret.push_back(a.p + v * x - RotateCCW90(v) * y);
    return ret;

//另一种求圆交点 //0无交点 1有1个 2有两个
int intersection_circle_circle(Point c1 , double r1 , Point c2 , double r2 , Point &p1 , Point &p2) {
    double d = Dis(c1 , c2);
    if(sgn(r1 + r2 - d) < 0) return 0;
    if(sgn(fabs(r1 - r2) - d) > 0) return 0;
    double a = atan2(c2.y - c1.y , c2.x - c1.x);
    double da = acos( (r1 * r1 + d * d - r2  *r2) / (2 * r1 * d));
    p1 = Point(cos(a - da),sin(a - da)) * r1 + c1;
    p2 = Point(cos(a + da),sin(a + da)) * r1 + c1;
    if (p1 == p2) return 1;
    return 2;
}
```

### 圆圆位置关系

```cpp
// 1:内含 2:内切 3:相交 4:外切 5:相离
int c_c_relation(const C& a, const C& v) {
    LD d = dist(a.p - v.p);
    if (sgn(d - a.r - v.r) > 0) return 5;
    if (sgn(d - a.r - v.r) == 0) return 4;
    LD l = fabs(a.r - v.r);
    if (sgn(d - l) > 0) return 3;
    if (sgn(d - l) == 0) return 2;
    if (sgn(d - l) < 0) return 1;
}
```

### 圆与多边形交

+ HDU 5130
+ 注意顺时针逆时针（可能要取绝对值）

```cpp
LD sector_area(const P& a, const P& b, LD r) {
    LD th = atan2(a.y, a.x) - atan2(b.y, b.x);
    while (th <= 0) th += 2 * PI;
    while (th > 2 * PI) th -= 2 * PI;
    th = min(th, 2 * PI - th);
    return r * r * th / 2;
}

LD c_tri_area(P a, P b, P center, LD r) {
    a = a - center; b = b - center;
    int ina = sgn(dist(a) - r) < 0, inb = sgn(dist(b) - r) < 0;
    if (ina && inb) {
        return fabs(cross(a, b)) / 2;
    } else {
        auto p = c_l_intersection(L(a, b), C(0, 0, r));
        if (ina ^ inb) {
            auto cr = p_on_seg(p[0], L(a, b)) ? p[0] : p[1];
            if (ina) return sector_area(b, cr, r) + fabs(cross(a, cr)) / 2;
            else return sector_area(a, cr, r) + fabs(cross(b, cr)) / 2;
        } else {
            if ((int) p.size() == 2 && p_on_seg(p[0], L(a, b))) {
                if (dist(p[0] - a) > dist(p[1] - a)) swap(p[0], p[1]);
                return sector_area(a, p[0], r) + sector_area(p[1], b, r)
                    + fabs(cross(p[0], p[1])) / 2;
            } else return sector_area(a, b, r);
        }
    }
}

typedef vector<P> S;
LD c_poly_area(S poly, const C& c) {
    LD ret = 0; int n = poly.size();
    FOR (i, 0, n) {
        int t = sgn(cross(poly[i] - c.p, poly[(i + 1) % n] - c.p));
        if (t) ret += t * c_tri_area(poly[i], poly[(i + 1) % n], c.p, c.r);
    }
    return ret;
}

//另一种
double det(Point a , Point b) {return a.x * b.y - a.y * b.x;}
double dot(Point a , Point b) {return a.x * b.x + a.y * b.y;}
Point operator * (Point a , double t) {return Point(a.x * t , a.y * t);}
Point operator + (Point a , Point b) {return Point(a.x + b.x , a.y + b.y);}
Point operator - (Point a , Point b) {return Point(a.x - b.x , a.y - b.y);}
double Length(Point a) {return sqrt(dot(a,a));}


double Tri_cir_insection(Circle C, Point A, Point B) {
    Point oa = A - C.c, ob = B - C.c;
    Point ba = A - B, bc = C.c - B;
    Point ab = B - A, ac = C.c - A;
    double doa = Length(oa), dob = Length(ob), dab = Length(ab), r = C.r;
    double x = (dot(ba, bc) + sqrt(r * r * dab * dab - det(ba, bc) * det(ba, bc))) / dab;
    double y = (dot(ab, ac) + sqrt(r * r * dab * dab - det(ab, ac) * det(ab, ac))) / dab;
    double ts = det(oa, ob) * 0.5;

    if (sgn(det(oa, ob)) == 0) return 0;
    if (sgn(doa - C.r) < 0 && sgn(dob - C.r) < 0) {
        return det(oa, ob) * 0.5;
    } else if (dob < r && doa >= r) //one in one out
    {
        return asin(ts * (1 - x / dab) * 2 / r / doa) * r * r * 0.5 + ts * x / dab;
    } else if (dob >= r && doa < r) // one out one in
    {
        return asin(ts * (1 - y / dab) * 2 / r / dob) * r * r * 0.5 + ts * y / dab;
    } else if (fabs(det(oa, ob)) >= r * dab || dot(ab, ac) <= 0 || dot(ba, bc) <= 0) // 只有弧
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
        return (asin(ts * (1 - x / dab) * 2 / r / doa) + asin(ts * (1 - y / dab) * 2 / r / dob)) * r * r * 0.5 +
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

### 圆的离散化、面积并

SPOJ: CIRU, EOJ: 284

+ 版本 1：复杂度 $O(n^3 \log n)$。虽然常数小，但还是难以接受。
+ 优点？想不出来。
+ 原理上是用竖线进行切分，然后对每一个切片分别计算。
+ 扫描线部分可以魔改，求各种东西。

```cpp
inline LD rt(LD x) { return sgn(x) == 0 ? 0 : sqrt(x); }
inline LD sq(LD x) { return x * x; }

// 圆弧
// 如果按照 x 离散化，圆弧是 "横着的"
// 记录圆弧的左端点、右端点、中点的坐标，和圆弧所在的圆
// 调用构造要保证 c.x - x.r <= xl < xr <= c.y + x.r
// t = 1 下圆弧 t = -1 上圆弧
struct CV {
    LD yl, yr, ym; C o; int type;
    CV() {}
    CV(LD yl, LD yr, LD ym, C c, int t)
        : yl(yl), yr(yr), ym(ym), type(t), o(c) {}
};

// 辅助函数 求圆上纵坐标
pair<LD, LD> c_point_eval(const C& c, LD x) {
    LD d = fabs(c.p.x - x), h = rt(sq(c.r) - sq(d));
    return {c.p.y - h, c.p.y + h};
}
// 构造上下圆弧
pair<CV, CV> pairwise_curves(const C& c, LD xl, LD xr) {
    LD yl1, yl2, yr1, yr2, ym1, ym2;
    tie(yl1, yl2) = c_point_eval(c, xl);
    tie(ym1, ym2) = c_point_eval(c, (xl + xr) / 2);
    tie(yr1, yr2) = c_point_eval(c, xr);
    return {CV(yl1, yr1, ym1, c, 1), CV(yl2, yr2, ym2, c, -1)};
}

// 离散化之后同一切片内的圆弧应该是不相交的
bool operator < (const CV& a, const CV& b) { return a.ym < b.ym; }
// 计算圆弧和连接圆弧端点的线段构成的封闭图形的面积
LD cv_area(const CV& v, LD xl, LD xr) {
    LD l = rt(sq(xr - xl) + sq(v.yr - v.yl));
    LD d = rt(sq(v.o.r) - sq(l / 2));
    LD ang = atan(l / d / 2);
    return ang * sq(v.o.r) - d * l / 2;
}

LD circle_union(const vector<C>& cs) {
    int n = cs.size();
    vector<LD> xs;
    FOR (i, 0, n) {
        xs.push_back(cs[i].p.x - cs[i].r);
        xs.push_back(cs[i].p.x);
        xs.push_back(cs[i].p.x + cs[i].r);
        FOR (j, i + 1, n) {
            auto pts = c_c_intersection(cs[i], cs[j]);
            for (auto& p: pts) xs.push_back(p.x);
        }
    }
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end(), [](LD x, LD y) { return sgn(x - y) == 0; }), xs.end());
    LD ans = 0;
    FOR (i, 0, (int) xs.size() - 1) {
        LD xl = xs[i], xr = xs[i + 1];
        vector<CV> intv;
        FOR (k, 0, n) {
            auto& c = cs[k];
            if (sgn(c.p.x - c.r - xl) <= 0 && sgn(c.p.x + c.r - xr) >= 0) {
                auto t = pairwise_curves(c, xl, xr);
                intv.push_back(t.first); intv.push_back(t.second);
            }
        }
        sort(intv.begin(), intv.end());

        vector<LD> areas(intv.size());
        FOR (i, 0, intv.size()) areas[i] = cv_area(intv[i], xl, xr);

        int cc = 0;
        FOR (i, 0, intv.size()) {
            if (cc > 0) {
                ans += (intv[i].yl - intv[i - 1].yl + intv[i].yr - intv[i - 1].yr) * (xr - xl) / 2;
                ans += intv[i - 1].type * areas[i - 1];
                ans -= intv[i].type * areas[i];
            }
            cc += intv[i].type;
        }
    }
    return ans;
}
```

+ 版本 2：复杂度 $O(n^2 \log n)$。
+ 原理是：认为所求部分是一个奇怪的多边形 + 若干弓形。然后对于每个圆分别求贡献的弓形，并累加多边形有向面积。
+ 同样可以魔改扫描线的部分，用于求周长、至少覆盖 $k$ 次等等。
+ 内含、内切、同一个圆的情况，通常需要特殊处理。
+ 下面的代码是 $k$ 圆覆盖。

```cpp
inline LD angle(const P& p) { return atan2(p.y, p.x); }

// 圆弧上的点
// p 是相对于圆心的坐标
// a 是在圆上的 atan2 [-PI, PI]
struct CP {
    P p; LD a; int t;
    CP() {}
    CP(P p, LD a, int t): p(p), a(a), t(t) {}
};
bool operator < (const CP& u, const CP& v) { return u.a < v.a; }
LD cv_area(LD r, const CP& q1, const CP& q2) {
    return (r * r * (q2.a - q1.a) - cross(q1.p, q2.p)) / 2;
}

LD ans[N];
void circle_union(const vector<C>& cs) {
    int n = cs.size();
    FOR (i, 0, n) {
        // 有相同的圆的话只考虑第一次出现
        bool ok = true;
        FOR (j, 0, i)
            if (sgn(cs[i].r - cs[j].r) == 0 && cs[i].p == cs[j].p) {
                ok = false;
                break;
            }
        if (!ok) continue;
        auto& c = cs[i];
        vector<CP> ev;
        int belong_to = 0;
        P bound = c.p + P(-c.r, 0);
        ev.emplace_back(bound, -PI, 0);
        ev.emplace_back(bound, PI, 0);
        FOR (j, 0, n) {
            if (i == j) continue;
            if (c_c_relation(c, cs[j]) <= 2) {
                if (sgn(cs[j].r - c.r) >= 0) // 完全被另一个圆包含，等于说叠了一层
                    belong_to++;
                continue;
            }
            auto its = c_c_intersection(c, cs[j]);
            if (its.size() == 2) {
                P p = its[1] - c.p, q = its[0] - c.p;
                LD a = angle(p), b = angle(q);
                if (sgn(a - b) > 0) {
                    ev.emplace_back(p, a, 1);
                    ev.emplace_back(bound, PI, -1);
                    ev.emplace_back(bound, -PI, 1);
                    ev.emplace_back(q, b, -1);
                } else {
                    ev.emplace_back(p, a, 1);
                    ev.emplace_back(q, b, -1);
                }
            }
        }
        sort(ev.begin(), ev.end());
        int cc = ev[0].t;
        FOR (j, 1, ev.size()) {
            int t = cc + belong_to;
            ans[t] += cross(ev[j - 1].p + c.p, ev[j].p + c.p) / 2;
            ans[t] += cv_area(c.r, ev[j - 1], ev[j]);
            cc += ev[j].t;
        }
    }
}
//另一种 k次面积交
#define sqr(x) ((x)*(x))
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

int CirCrossCir(cp p1, double r1, cp p2, double r2, cp &cp1, cp &cp2) {
    double mx = p2.x - p1.x, sx = p2.x + p1.x, mx2 = mx * mx;
    double my = p2.y - p1.y, sy = p2.y + p1.y, my2 = my * my;
    double sq = mx2 + my2, d = -(sq - sqr(r1 - r2)) * (sq - sqr(r1 + r2));
    if (d + eps < 0) return 0;
    if (d < eps) d = 0;
    else d = sqrt(d);
    double x = mx * ((r1 + r2) * (r1 - r2) + mx * sx) + sx * my2;
    double y = my * ((r1 + r2) * (r1 - r2) + my * sy) + sy * mx2;
    double dx = mx * d, dy = my * d;
    sq *= 2;
    cp1.x = (x - dy) / sq;
    cp1.y = (y + dx) / sq;
    cp2.x = (x + dy) / sq;
    cp2.y = (y - dx) / sq;
    if (d > eps) return 2;
    else return 1;
}

bool circmp(const cp &u, const cp &v) {
    return sgn(u.r - v.r) < 0;
}

bool cmp(const cp &u, const cp &v) {
    if (sgn(u.angle - v.angle))
        return u.angle < v.angle;
    return u.d > v.d;
}

double calc(cp cir, cp cp1, cp cp2) {
    double ans = (cp2.angle - cp1.angle) * sqr(cir.r)
                 - Cross(cir, cp1, cp2) + Cross(cp(0, 0), cp1, cp2);
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
            if (CirCrossCir(cir[i], cir[i].r, cir[j], cir[j].r,
                            cp2, cp1) < 2)
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

### 最小圆覆盖

+ 随机增量。期望复杂度 $O(n)$。

```cpp
P compute_circle_center(P a, P b) { return (a + b) / 2; }
bool p_in_circle(const P& p, const C& c) {
    return sgn(dist(p - c.p) - c.r) <= 0;
}
C min_circle_cover(const vector<P> &in) {
    vector<P> a(in.begin(), in.end());

    random_device rd; // c++ 14
    mt19937 g(rd());
    shuffle(a.begin(), a.end(), g);
    // random_shuffle(a.begin(), a.end());

    P c = a[0]; LD r = 0; int n = a.size();
    FOR (i, 1, n) if (!p_in_circle(a[i], {c, r})) {
        c = a[i]; r = 0;
        FOR (j, 0, i) if (!p_in_circle(a[j], {c, r})) {
            c = compute_circle_center(a[i], a[j]);
            r = dist(a[j] - c);
            FOR (k, 0, j) if (!p_in_circle(a[k], {c, r})) {
                c = compute_circle_center(a[i], a[j], a[k]);
                r = dist(a[k] - c);
            }
        }
    }
    return {c, r};
}

//另一种 最小圆覆盖
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

### 圆的反演

```cpp
C inv(C c, const P& o) {
    LD d = dist(c.p - o);
    assert(sgn(d) != 0);
    LD a = 1 / (d - c.r);
    LD b = 1 / (d + c.r);
    c.r = (a - b) / 2 * R2;
    c.p = o + (c.p - o) * ((a + b) * R2 / 2 / d);
    return c;
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

## 三维计算几何

```cpp
struct P;
struct L;
typedef P V;

struct P {
    LD x, y, z;
    explicit P(LD x = 0, LD y = 0, LD z = 0): x(x), y(y), z(z) {}
    explicit P(const L& l);
};

struct L {
    P s, t;
    L() {}
    L(P s, P t): s(s), t(t) {}
};

struct F {
    P a, b, c;
    F() {}
    F(P a, P b, P c): a(a), b(b), c(c) {}
};

P operator + (const P& a, const P& b) { return P(a.x + b.x, a.y + b.y, a.z + b.z); }
P operator - (const P& a, const P& b) { return P(a.x - b.x, a.y - b.y, a.z - b.z); }
P operator * (const P& a, LD k) { return P(a.x * k, a.y * k, a.z * k); }
P operator / (const P& a, LD k) { return P(a.x / k, a.y / k, a.z / k); }
inline int operator < (const P& a, const P& b) {
    return sgn(a.x - b.x) < 0 || (sgn(a.x - b.x) == 0 && (sgn(a.y - b.y) < 0 ||
                                  (sgn(a.y - b.y) == 0 && sgn(a.z - b.z) < 0)));
}
bool operator == (const P& a, const P& b) { return !sgn(a.x - b.x) && !sgn(a.y - b.y) && !sgn(a.z - b.z); }
P::P(const L& l) { *this = l.t - l.s; }
ostream &operator << (ostream &os, const P &p) {
    return (os << "(" << p.x << "," << p.y << "," << p.z << ")");
}
istream &operator >> (istream &is, P &p) {
    return (is >> p.x >> p.y >> p.z);
}

// --------------------------------------------
LD dist2(const P& p) { return p.x * p.x + p.y * p.y + p.z * p.z; }
LD dist(const P& p) { return sqrt(dist2(p)); }
LD dot(const V& a, const V& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
P cross(const P& v, const P& w) {
    return P(v.y * w.z - v.z * w.y, v.z * w.x - v.x * w.z, v.x * w.y - v.y * w.x);
}
LD mix(const V& a, const V& b, const V& c) { return dot(a, cross(b, c)); }
```

### 旋转

```cpp
// 逆时针旋转 r 弧度
// axis = 0 绕 x 轴
// axis = 1 绕 y 轴
// axis = 2 绕 z 轴
P rotation(const P& p, const LD& r, int axis = 0) {
    if (axis == 0)
        return P(p.x, p.y * cos(r) - p.z * sin(r), p.y * sin(r) + p.z * cos(r));
    else if (axis == 1)
        return P(p.z * cos(r) - p.x * sin(r), p.y, p.z * sin(r) + p.x * cos(r));
    else if (axis == 2)
        return P(p.x * cos(r) - p.y * sin(r), p.x * sin(r) + p.y * cos(r), p.z);
}
// n 是单位向量 表示旋转轴
// 模板是顺时针的
P rotation(const P& p, const LD& r, const P& n) {
    LD c = cos(r), s = sin(r), x = n.x, y = n.y, z = n.z;
    return P((x * x * (1 - c) + c) * p.x + (x * y * (1 - c) + z * s) * p.y + (x * z * (1 - c) - y * s) * p.z,
             (x * y * (1 - c) - z * s) * p.x + (y * y * (1 - c) + c) * p.y + (y * z * (1 - c) + x * s) * p.z,
             (x * z * (1 - c) + y * s) * p.x + (y * z * (1 - c) - x * s) * p.y + (z * z * (1 - c) + c) * p.z);
}
```

### 线、面

函数相互依赖，所以交织在一起了。

```cpp
// 点在线段上  <= 0包含端点 < 0 则不包含
bool p_on_seg(const P& p, const L& seg) {
    P a = seg.s, b = seg.t;
    return !sgn(dist2(cross(p - a, b - a))) && sgn(dot(p - a, p - b)) <= 0;
}
// 点到直线距离
LD dist_to_line(const P& p, const L& l) {
    return dist(cross(l.s - p, l.t - p)) / dist(l);
}
// 点到线段距离
LD dist_to_seg(const P& p, const L& l) {
    if (l.s == l.t) return dist(p - l.s);
    V vs = p - l.s, vt = p - l.t;
    if (sgn(dot(l, vs)) < 0) return dist(vs);
    else if (sgn(dot(l, vt)) > 0) return dist(vt);
    else return dist_to_line(p, l);
}

P norm(const F& f) { return cross(f.a - f.b, f.b - f.c); }
int p_on_plane(const F& f, const P& p) { return sgn(dot(norm(f), p - f.a)) == 0; }

// 判两点在线段异侧 点在线段上返回 0 不共面无意义
int opposite_side(const P& u, const P& v, const L& l) {
    return sgn(dot(cross(P(l), u - l.s), cross(P(l), v - l.s))) < 0;
}

bool parallel(const L& a, const L& b) { return !sgn(dist2(cross(P(a), P(b)))); }
// 线段相交
int s_intersect(const L& u, const L& v) {
    return p_on_plane(F(u.s, u.t, v.s), v.t) && 
           opposite_side(u.s, u.t, v) &&
           opposite_side(v.s, v.t, u);
}
```

### 凸包

增量法。先将所有的点打乱顺序，然后选择四个不共面的点组成一个四面体，如果找不到说明凸包不存在。然后遍历剩余的点，不断更新凸包。对遍历到的点做如下处理。

1. 如果点在凸包内，则不更新。
2. 如果点在凸包外，那么找到所有原凸包上所有分隔了对于这个点可见面和不可见面的边，以这样的边的两个点和新的点创建新的面加入凸包中。

```cpp
struct FT {
    int a, b, c;
    FT() { }
    FT(int a, int b, int c) : a(a), b(b), c(c) { }
};

bool p_on_line(const P& p, const L& l) {
    return !sgn(dist2(cross(p - l.s, P(l))));
}

vector<F> convex_hull(vector<P> &p) {
    sort(p.begin(), p.end());
    p.erase(unique(p.begin(), p.end()), p.end());

    random_device rd; // c++ 14
    mt19937 g(rd());
    shuffle(p.begin(), p.end(), g);
    // random_shuffle(p.begin(), p.end());

    vector<FT> face;
    FOR (i, 2, p.size()) {
        if (p_on_line(p[i], L(p[0], p[1]))) continue;
        swap(p[i], p[2]);
        FOR (j, i + 1, p.size())
            if (sgn(mix(p[1] - p[0], p[2] - p[1], p[j] - p[0]))) {
                swap(p[j], p[3]);
                face.emplace_back(0, 1, 2);
                face.emplace_back(0, 2, 1);
                goto found;
            }
    }
found:
    vector<vector<int>> mk(p.size(), vector<int>(p.size()));
    FOR (v, 3, p.size()) {
        vector<FT> tmp;
        FOR (i, 0, face.size()) {
            int a = face[i].a, b = face[i].b, c = face[i].c;
            if (sgn(mix(p[a] - p[v], p[b] - p[v], p[c] - p[v])) < 0) {
                mk[a][b] = mk[b][a] = v;
                mk[b][c] = mk[c][b] = v;
                mk[c][a] = mk[a][c] = v;
            } else tmp.push_back(face[i]);
        }
        face = tmp;
        FOR (i, 0, tmp.size()) {
            int a = face[i].a, b = face[i].b, c = face[i].c;
            if (mk[a][b] == v) face.emplace_back(b, a, v);
            if (mk[b][c] == v) face.emplace_back(c, b, v);
            if (mk[c][a] == v) face.emplace_back(a, c, v);
        }
    }
    vector<F> out;
    FOR (i, 0, face.size())
        out.emplace_back(p[face[i].a], p[face[i].b], p[face[i].c]);
    return out;
}
```

### 最小球覆盖

```cpp
struct Point {
    double x, y, z;

    Point(double X = 0, double Y = 0, double Z = 0) {
        x = X;
        y = Y;
        z = Z;
    }
} p[maxn];

inline double Dis(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

int n;

double Solve() {
    double Step = 200, ans = 1e9, mt;
    Point z = Point(0.0, 0.0, 0.0);
    int s = 0;
    while (Step > eps) {
        for (int i = 1; i <= n; ++i) {
            if (Dis(z, p[s]) < Dis(z, p[i]))
                s = i;
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

```cpp
struct Point {
    double x, y, z;

    Point(double X = 0, double Y = 0, double Z = 0) {
        x = X;
        y = Y;
        z = Z;
    }
} p[maxn];

inline double Dis(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

int n;

double Solve() {
    double Step = 200, ans = 1e9, mt;
    Point z = Point(0.0, 0.0, 0.0);
    int s = 0;
    while (Step > eps) {
        for (int i = 1; i <= n; ++i) {
            if (Dis(z, p[s]) < Dis(z, p[i]))
                s = i;
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

