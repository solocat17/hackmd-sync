---
title: 解題紀錄
---

<!-- latex cheatsheet https://hackmd.io/@sysprog/B1RwlM85Z?type=view 
katex cheatsheet
https://katex.org/docs/supported.html
-->

---

## [AtCoder DP contest](https://atcoder.jp/contests/dp/tasks)

[TOC]

----

### pA

[link](https://atcoder.jp/contests/dp/tasks/dp_a)

**題目：$N$顆依序編號$0, 2,\cdots ,N-1$的石頭具有高度$h_i$，一隻青蛙在第$i$顆石頭時可以跳到$i + 1$或$i + 2$顆石頭，並花費$|h_i - h_j|$，試求從第$0$顆石頭跳到第$N$顆石頭的最小花費**

- dp轉移式：
- \begin{equation}
  dp[i]=\begin{cases}
    0, &\text{if $i = 0$}.\\
    |h_0-h_1|, &\text{if $i = 1$}.\\
    \min(dp[i-1]+|h_i-h_{i-1}|,\\\quad \quad \ dp[i-2]+|h_i-h_{i-2}|), &\text{otherwise}.
  \end{cases}
\end{equation}
- 答案：$dp[N-1]$
- 複雜度：時間$\mathcal O(N)$，空間$\mathcal O(N)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;
 
int h[100000], dp[100000];
 
signed main() {
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) cin >> h[i];
  dp[0] = 0;
  dp[1] = abs(h[0] - h[1]);
  for (int i = 2; i < n; i++) {
    dp[i] = min(dp[i - 1] + abs(h[i] - h[i - 1]),
                dp[i - 2] + abs(h[i] - h[i - 2]));
  }
  cout << dp[n - 1] << endl;
  return 0;
}
```
:::

----

### pB

[link](https://atcoder.jp/contests/dp/tasks/dp_b)

**題目：和前一題一樣，差在他可以跳到距離為$K$的石頭**

- dp轉移式：
- \begin{equation}
  dp[i]=\begin{cases}
    0, &\text{if $i = 0$}.\\
    \min(dp[i-1]+|h_i-h_{i-1}|,\\\quad \quad \ dp[i-2]+|h_i-h_{i-2}|,\\\quad \quad \quad \quad \quad \quad \cdots\\\quad \quad \ dp[i-K] + |h_i-h_{i-K}|), &\text{otherwise}.
  \end{cases}
\end{equation}
- 答案：$dp[N-1]$
- 複雜度：時間$\mathcal O(NK)$，空間$\mathcal O(N)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;
const int INF = 0x3f3f3f3f;
 
int h[100000], dp[100000];
 
signed main() {
  int n, k;
  cin >> n >> k;
  cin >> h[0];
  dp[0] = 0;
  for (int i = 1; i < n; i++) {
    cin >> h[i];
    dp[i] = INF;
    for (int j = max(0, i - k); j < i; j++) {
      dp[i] = min(dp[i], dp[j] + abs(h[j] - h[i]));
    }
  }
  cout << dp[n - 1] << endl;
  return 0;
}
```
:::

----

### pC

[link](https://atcoder.jp/contests/dp/tasks/dp_c)

**題目：$N$天內每天有三件事情可以選一件做，在第$i$天做事件$a, b, c$分別可以獲得$a_i, b_i, c_i$的喜悅感，但她沒辦法連續兩天做同一件事，試求喜悅感的最大和**

- dp轉移式：
- \begin{equation}
  dp[i][j]=\begin{cases}
    j_i, &\text{if i = 0}.\\
    dp[i-1][j] +\\
    \max_{k\in \{range[0,3)|k\neq j\}}dp[i][k], &\text{otherwise}.
  \end{cases}
\end{equation}
:::info
寫這副德性還不如直接去看code
:::
- 答案：$\max_{i\in [0,3)}dp[n-1][i]$
- 複雜度：時間$\mathcal O(N)$，空間$\mathcal O(N)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;
 
int abc[100000][3], dp[100000][3];
 
signed main() {
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) cin >> abc[i][0] >> abc[i][1] >> abc[i][2];

  dp[0][0] = abc[0][0];
  dp[0][1] = abc[0][1];
  dp[0][2] = abc[0][2];
  for (int i = 1; i < n; i++) {
    dp[i][0] = abc[i][0] + max(dp[i - 1][1], dp[i - 1][2]);
    dp[i][1] = abc[i][1] + max(dp[i - 1][0], dp[i - 1][2]);
    dp[i][2] = abc[i][2] + max(dp[i - 1][0], dp[i - 1][1]);
  }
  cout << *max_element(dp[n - 1], dp[n - 1] + 3) << endl;
  return 0;
}
```
:::

----

### pD

[link](https://atcoder.jp/contests/dp/tasks/dp_d)

**題目：背包問題：有$N$個物品，第$i$物具有$w_i$的重量與$v_i(1\leq v_i\leq 10^9)$的價值，我們有一個容量為$W(1\leq W\leq 10^5)$的包包，在不超過容量上限下，求背包能裝下的最大值**

#### 解法一

- 考慮兩個維度，第一個維度放「目前拿到哪個物品」，第二個維度放「拿到現在的總重」，裡面維護價值最大值
- dp轉移式：
- \begin{equation}
  dp[i][j]=\begin{cases}
    dp[i-1][j],&\text{if $w_i>j$}.\\
    \max(dp[i-1][j],dp[i][j-w_i]+v_i),&\text{otherwise}.
  \end{cases}
\end{equation}
- 答案：$dp[N][W]$
- 複雜度：時間$\mathcal O(NW)$，空間$\mathcal O(NW)$

::: spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;
#define int long long
 
int dp[105][100005], w[105], v[105];
 
signed main() {
  int n, cap;
  cin >> n >> cap;
  for (int i = 1; i <= n; i++) cin >> w[i] >> v[i];

  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= cap; j++) {
      if (j - w[i] >= 0) dp[i][j] = max(dp[i - 1][j],
                                        dp[i - 1][j - w[i]] + v[i]);
      else dp[i][j] = dp[i - 1][j];
    }
  }
  cout << dp[n][cap] << endl;
  return 0;
}
```
:::

#### 解法二

- 考慮一維維護目前重量下的最大價值，每次知道下一個物品價值及重量時不斷更新
- dp轉移式：
- $$dp[j]=\max(dp[j], dp[j-w_i]+v_i)$$
- 答案：$dp[W]$
- 複雜度：時間$\mathcal O(NW)$，空間$\mathcal O(W)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;
#define int long long
 
int dp[100005], w, v;
 
signed main() {
  int n, cap;
  cin >> n >> cap;

  for (int i = 1; i <= n; i++) {
    cin >> w >> v;
    for (int j = cap; j >= w; j--) {
      dp[j] = max(dp[j - w] + v, dp[j]);
    }
  }
  cout << dp[cap] << endl;
  return 0;
}
```
:::

----

### pE

[link](https://atcoder.jp/contests/dp/tasks/dp_e)

**題目：和上題基本上一樣，差在值域**
\begin{cases}
1\leq N\leq 100\\
1\leq W\leq 10^5\\
1\leq w_i\leq W\\
1\leq v_i\leq 10^9
\end{cases}$$\Downarrow$$\begin{cases}
1\leq N\leq 100\\
1\leq W\leq 10^9\\
1\leq w_i\leq W\\
1\leq v_i\leq 10^3
\end{cases}

#### 解法一

- 一樣考慮兩個維度，一維放「目前拿到哪個物品」，另一維放「拿到現在的總價值」，裡面維護重量最小值
- dp轉移式：
- \begin{equation}
dp[i][j]=\begin{cases}
dp[i-1][j],&\text{if $v_i < j$}\\
\min(dp[i-1][j],dp[i-1][j-v_i]+w_i),&\text{otherwise}
\end{cases}
\end{equation}
- 答案：$\max_{i\in range[1, 100001)} dp[n][i]$
- 複雜度：時間$\mathcal O(N\sum v)$，空間$\mathcal O(N\sum v)$
:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;
const int INF = 0x3f3f3f3f;
 
int dp[105][100005], w[100005], v[100005];
 
signed main() {
  int n, cap;
  cin >> n >> cap;

  for (int i = 1; i <= n; i++) cin >> w[i] >> v[i];
  fill(&dp[0][0], &dp[n][100005], INF);

  dp[1][0] = 0; dp[1][v[1]] = w[1];
  for (int i = 2; i <= n; i++) {
    for (int j = 0; j <= 100000; j++) {
      if (v[i] <= j){
        dp[i][j] = min(dp[i - 1][j],
                       dp[i - 1][j - v[i]] + w[i]);
      }
      else dp[i][j] = dp[i - 1][j];
    }
  }
  int mx = -INF;
  for (int i = 1; i <= 100000; i++) {
    if (dp[n][i] <= cap) {
      mx = max(mx, i);
    }
  }
  cout << mx << endl;
  return 0;
}
```
:::

#### 解法二

- 和前一題一樣考慮只算一個維度，並同時不斷更新答案
- dp轉移式
- $$dp[j] = \min(dp[j], dp[j-v_i] + w_i)$$
- 答案：$\max_{i\in range[1, 100001)} dp[i]$
- 複雜度：時間$\mathcal O(N\sum v)$，空間$\mathcal O(\sum v)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;
const int INF = 0x3f3f3f3f;
 
int dp[100005], w, v;
 
signed main() {
  int n, cap;
  cin >> n >> cap;

  fill(dp, dp + 100005, INF);
  dp[0] = 0;

  for (int i = 1; i <= n; i++) {
    cin >> w >> v;
    for (int j = 100000; j >= v; j--) {
      dp[j] = min(dp[j], dp[j - v] + w);  
    }
  }
  int mx = -INF;
  for (int i = 1; i <= 100000; i++) {
    if (dp[i] <= cap) {
      mx = max(mx, i);
    }
  }
  cout << mx << endl;
  return 0;
}
```
:::

----

### pF

[link](https://atcoder.jp/contests/dp/tasks/dp_f)

**題目：給你兩個字串$s, t$，請找出他們的最長共同子字串**

- 考慮蓋表，$dp[i][j]$表示$s_0$到$s_{i-1}$，$t_0$到$t_{j-1}$為止最長共同子字串的長度
- dp轉移式：
- \begin{equation}
    dp[i][j]=
    \begin{cases}
      dp[i-1][j-1]+1,&\text{if $s_{i-1} = t_{j-1}$}\\
      \max(dp[i-1][j], dp[i][j-1]),&\text{otherwise}
    \end{cases}
  \end{equation}
- 答案：回朔找解，從$dp[|s|][|t|]$往回找，如果$s_{i-1}=t_{j-1}$，把答案字串加上$s_{i-1}$並把`i--; j--;`；如果不一樣則按照蓋表的順序，已知我們是從上面或左邊比較大的那個來的，我們就往比較大的那邊回朔
- 複雜度：時間$\mathcal O(|s||t|)$，空間$\mathcal O(|s||t|)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;


int dp[3005][3005], i, j, slen, tlen;


signed main() {
  string s, t;
  cin >> s >> t;
  slen = s.size(), tlen = t.size();
  for (i = 0; i <= slen; i++) {
    for (j = 0; j <= tlen; j++) {
      if (!i || !j) dp[i][j] = 0;

      else if (s[i - 1] == t[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
      else dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  i = slen, j = tlen;
  string ans;
  while (i && j) {
    if (s[i - 1] == t[j - 1]) {
      ans += s[i - 1];
      i--; j--;
    }

    else if (dp[i - 1][j] > dp[i][j - 1]) i--;
    else j--;
  }
  reverse(ans.begin(), ans.end());
  cout << ans << endl;
  return 0;
}
```
:::

----

### pG

[link](https://atcoder.jp/contests/dp/tasks/dp_g)

**題目：給你一張DAG，求圖上的最長路徑**

#### 解法一

- 做拓樸排序(Kahn's algorithm)，從最前面的點開始算最長路徑，每次做到一個點時更新所有他能到的點的答案
- dp轉移式：
- $$dp[v] = \max_{u \in \deg^-(v)} (dp[u]+1)$$
- 答案：在更新過程中不斷更新最長距離$ans=\max dp[i]$
- 複雜度：時間$\mathcal O(N+M)$，空間$\mathcal O(NM)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;


signed main() {
  int n, m, s, t;
  cin >> n >> m;
  vector<vector<int>> graph(n + 1);
  vector<int> indeg(n + 1, 0);
  for (int i = 0; i < m; i++) {
    cin >> s >> t;
    graph[s].emplace_back(t);
    indeg[t]++;
  }


  queue<int> q;
  for (int i = 1; i <= n; i++) {
    if (!indeg[i]) q.push(i);
  }

  vector<int> mxdis(n + 1);
  int mx = -1;
  while (q.size()) {
    for (int cur : graph[q.front()]) {
      mxdis[cur] = max(mxdis[cur], mxdis[q.front()] + 1);
      mx = max(mx, mxdis[cur]);
      indeg[cur]--;
      if(!indeg[cur]) q.push(cur);
    }
    q.pop();
  }

  cout << mx << endl;
  return 0;
}
```
:::


#### 解法二

- 其實也不見得要先找到最前面的點，可以考慮從任意點開始做dfs，不斷找他的入度的最長路徑同時更新這個點的答案

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;


vector<int> graph[100005];  //from where -> reverse edge
int mxdis[100005];


int dfs(int cur) {
  if (graph[cur].empty()) return mxdis[cur] = 0;
  if (mxdis[cur] != -1) return mxdis[cur];

  for(int i : graph[cur]){
    mxdis[cur] = max(mxdis[cur], dfs(i));
  }
  mxdis[cur]++;
  return mxdis[cur];
}


signed main() {
  int n, m, s, t;
  cin >> n >> m;
  fill(mxdis, mxdis + n + 1, -1);
  for (int i = 0; i < m; i++) {
    cin >> s >> t;
    graph[t].emplace_back(s);
  }

  int mx = -1;

  for (int i = 1; i <= n; i++) mx = max(mx, dfs(i));

  cout << mx << endl;
  return 0;
}
```
:::

----

### pH

[link](https://atcoder.jp/contests/dp/tasks/dp_h)

**題目：給你一H\*W由'#'及'.'組成的矩陣，其中'.'可走#'不可走，求從(1,1)走到(H,W)的方法數**

- 小技巧是按照題目說的從(1,1)開始存值，x或y如果是0就把它值設成0，好寫很多(?)
- dp轉移式
- \begin{equation}
    dp[i][j]=
    \begin{cases}
      0,&\text{if $i = 0$ or $j = 0$}\\
      1,&\text{if $i = 1$ and $j = 1$}\\
      dp[i-1][j]+dp[i][j-1],&\text{otherwise}
    \end{cases}
  \end{equation}
- 答案：$dp[H][W]$
- 複雜度：時間$\mathcal O(HW)$，空間$\mathcal O(HW)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;
const int MOD = 1e9+7;


char mp[1005][1005];
int dp[1005][1005];


signed main() {
  int h, w;
  cin >> h >> w;
  for (int i = 1; i <= h; i++) {
    for (int j = 1; j <= w; j++) cin >> mp[i][j];
  }

  for (int i = 0; i < 1005; i++) {
    dp[0][i] = dp[i][0] = 0;
  }
  dp[1][1] = 1;

  for (int i = 1; i <= h; i++) {
    for (int j = 1; j <= w; j++) {
      if (i == 1 && j == 1) continue;
      if (mp[i][j] == '#') dp[i][j] = 0;
      else dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) % MOD;
    }
  }
  cout << dp[h][w] << endl;
  return 0;
}
```
:::

----

### pI

[link](https://atcoder.jp/contests/dp/tasks/dp_i)

**題目：有$N$枚硬幣，第$i$枚硬幣擲出人頭的機率為$p_i$，試求全部擲出人頭比數字多的機率**

- 考慮兩個維度，第一個維度放當前做到第幾個硬幣，第二個維度維護當前有幾個硬幣是人頭面
- dp轉移式：
- \begin{equation}
    dp[i][j]=
    \begin{cases}
      dp[i−1][j]\times(1−p_i),&\text{if j = 0}\\
      dp[i-1][j]\times(1-p_i)+dp[i-1][j-1]\times p_i,&\text{otherwise}
    \end{cases}
  \end{equation}
- 答案：$\sum_{i=\frac{N+1}{2}}^{n} dp[N][i]$
- 複雜度：時間$\mathcal O(N^2)$，空間$\mathcal O(N^2)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;


double dp[3005][3005], p[3005];


signed main() {
  int n;
  cin >> n;
  for (int i = 1; i <= n; i++) cin >> p[i];

  dp[1][0] = 1 - p[1];
  dp[1][1] = p[1];

  for (int i = 2; i <= n; i++) {
    for (int j = 0; j <= i; j++) {
      if (j == 0) dp[i][j] = dp[i - 1][j] * (1 - p[i]);
      else dp[i][j] = dp[i - 1][j - 1] * p[i] + dp[i - 1][j] * (1 - p[i]);
    }
  }

  double ans = 0.0000000000;
  for (int i = (n + 1) / 2; i <= n; i++) ans += dp[n][i];

  cout << fixed << setprecision(10) << ans << endl;
  return 0;
}
```
:::

----

### pJ

[link](https://atcoder.jp/contests/dp/tasks/dp_j)

**題目：有$N$個碟子，第$i$個碟子上有$a_i(1\leq a_i\leq 3)$塊壽司，每次操作隨機從$1,2,3,\cdots ,N$個碟子中選一個吃掉碟子上的一塊壽司，若選中的碟子上沒有壽司則不做任何動作。試求將所有壽司吃完所需操作數的期望值**

- 考慮三個維度，第$i(1\leq i\leq 3)$維維護有幾個碟子上有$i$塊壽司
- dp轉移式：
- $$dp[i][j][k]=\frac{n}{i+j+k}\\+\frac{i}{i+j+k}\times dp[i-1][j][k]\\+\frac{j}{i+j+k}\times dp[i+1][j-1][k]\\+\frac{k}{i+j+k}\times dp[i][j+1][k-1]$$
- 答案：$dp[count(a_i=1)][count(a_i=2)][count(a_i=3)]$
- 複雜度：時間$\mathcal O(N^3)$，空間$\mathcal O(N^3)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;


double dp[305][305][305];
int n, tmp, cnt[3];
 
 
double rec(int i, int j, int k) {
  if (i == 0 && j == 0 && k == 0) return dp[i][j][k] = 0;
  if (dp[i][j][k] >= 0) return dp[i][j][k];

  double sum = i + j + k;
  dp[i][j][k] = n / sum;
  if (i) dp[i][j][k] += (rec(i - 1, j, k) * i / sum);
  if (j) dp[i][j][k] += (rec(i + 1, j - 1, k) * j / sum);
  if (k) dp[i][j][k] += (rec(i, j + 1, k - 1) * k / sum);

  return dp[i][j][k];
}


signed main() {
  memset(dp, -1, sizeof(dp));
  cin >> n;
  cnt[0] = cnt[1] = cnt[2] = 0;
  for (int i = 0; i < n; i++) {
    cin >> tmp;
    tmp--;
    cnt[tmp]++;
  }
  cout << fixed << setprecision(10) << rec(cnt[0], cnt[1], cnt[2]) << endl;
  return 0;
}
```
:::

----

### pK

[link](https://atcoder.jp/contests/dp/tasks/dp_k)

**題目：給定一大小為$N$的集合$a$以及一個包含$K$顆石頭的石堆，兩人輪流進行操作，每次操作可以選擇$a$中的一個數字$x$並從石堆中取走$x$顆石頭，取不走的人就輸了，求贏家是誰**

- $dp[i]$放在剩下$i$顆石頭時，先手是否能必勝，則考慮在剩下$i$顆石頭時，如果拿走$a_j$顆石頭可以使後手必輸，則先手必勝
- dp轉移式：
- $$dp[i]=(\sum_{j=0}^{a_j\leq i}(!dp[i-a_j]) >0)$$
- 答案：$dp[K]$
- 複雜度：時間$\mathcal O(NK)$，空間$\mathcal O(K)$

:::spoiler AC Code
```cpp=
#include <bits/stdc++.h>
using namespace std;


int a[105];
bool dp[100005] = {0};


signed main() {
  int n, k;
  cin >> n >> k;
  for (int i = 0; i < n; i++) cin >> a[i];

  for (int i = 0; i <= k; i++) {
    for (int j = 0; j < n && a[j] <= i; j++) {
      dp[i] |= !dp[i - a[j]];
    }
  }
  cout << (dp[k] ? "First\n" : "Second\n");
  return 0;
}
```
:::

----

### pL

[link](https://atcoder.jp/contests/dp/tasks/dp_l)

**題目：給一個序列$a$，每次可以從最前面或最後面取走一個元素$x$並得到$x$分，雙方都想最大化與對方的得分差，問最後先手得分-後手得分為多少**

- 先想想三個維度的做法：考慮$dp[i][j][k]$表示在原序列只剩下$a_i$到$a_j$時，輪到$k$拿的最大得分差
- dp轉移式：
- $$dp[i][j][0]=\max(a_i+dp[i+1][j][1],a_j+dp[i][j-1][1])\\
  dp[i][j][1]=\min(a_i+dp[i+1][j][0],a_j+dp[i][j-1][0])$$
- 意思是指在輪到$[k]$拿且區間剩下$[i, j)$時，他會希望拿到最大化的分差，分差的值就是較大的(上個人的最大化和當前能拿的最大值)
- ***可以優化成兩個維度的做法***
- 考慮每個人都只會在輪到自己時能最大化/最小化當前答案，因此可以理解成對於每次最大化的操作，上次操作必是在進行最小化
- 重列dp轉移式：
- $$dp[i][j]=max(a_i-dp[i+1][j],a_j-dp[i][j-1])$$
- 答案：$dp[0][n]$
- 複雜度：時間$\mathcal O(N^2)$，空間$\mathcal O(N^2)$

----

### pM

[link](https://atcoder.jp/contests/dp/tasks/dp_m)

**題目：有$N$個小孩以及$K$顆糖果，每個小孩可以接受$0$到$a_i$顆糖，糖果不能有剩，問有幾種分配方法$(mod 10^9+7)$**

----

### pN

[link](https://atcoder.jp/contests/dp/tasks/dp_n)

**題目：$N$隻史萊姆站一排，每次操作可以把相鄰兩隻尺寸分別為$a,b$的史萊姆合併成一隻尺寸$a+b$的史萊姆，並花費$a+b$，試求將所有史萊姆合併成一隻的最小花費**

#### 解法一
- 區間dp經典題，$\mathcal O(N^3)$的做法比較顯然：$dp[i][j]$放合併$i$到$j$所有史萊姆的最小花費，狀態$\mathcal O(N^2)$種，轉移最長$O(N)$
- dp轉移式：
- $$dp[i][j]=\min_{k\in range[i,j)}(dp[i][k] + dp[k+1][j] + s[i][j])$$
- 答案：$dp[1][n]$
- 複雜度：時間$\mathcal O(N^3)$，空間$\mathcal O(N^2)$

#### 解法二

- 定義一個$[l,r)$區間的最優合併點$m$指在所有合併$[l,r)$區間的方法中，合併$[l,m)$以及$[m,r)$兩區間可得最小花費
- 觀察之後可以發現對於相同左界的許多區間，**右界較大的區間最優合併點必定比較右邊**；同理對於相同右界的許多區間，左界較小的區間最優合併點比較左邊。
- 因此對於三個區間$[i,k),[i,l),[j,l)\ \ (i<j<k<l)$而言，如果$[i,k)$的最優合併點在$[i,j)$中的某處且$[j,l)$的最優合併點在$[k,l)$中的某處，則$[i,l)$的最佳合併點必會在$[j,k)$中的某處
