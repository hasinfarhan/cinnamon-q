#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

long long get_sum(const vector<long long>& a) {
    long long sum = 0;
    for (long long val : a) {
        sum += val;
    }
    return sum;
}

void solve() {
    int n;
    long long k;
    cin >> n >> k;
    vector<long long> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }
    sort(a.begin(), a.end());

    long long initial_sum = get_sum(a);
    long long max_total_sum = initial_sum;

    vector<long long> prefix_sum(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        prefix_sum[i + 1] = prefix_sum[i] + a[i];
    }

    for (int x = 1; x <= n; x += 2) {
        long long median_val = a[(x - 1) / 2];
        long long sub_sum = prefix_sum[x];
        long long cost_per_op = (long long)x * median_val - sub_sum;

        if (cost_per_op <= 0) {
             max_total_sum = max(max_total_sum, initial_sum);
             continue;
        }

        long long num_ops = k / cost_per_op;
        long long remaining_k = k % cost_per_op;

        long long current_sum = initial_sum + num_ops * cost_per_op;

        // With remaining k, we can potentially increase the smallest num_ops*x + 1 elements
        long long num_elements_to_increase = num_ops * x + 1;
        if (num_elements_to_increase <= n) {
            current_sum += remaining_k / num_elements_to_increase;
        }
        
        max_total_sum = max(max_total_sum, current_sum);
    }

    cout << max_total_sum << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int t;
    cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}
