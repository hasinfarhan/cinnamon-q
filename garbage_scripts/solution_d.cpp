#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

struct Segment {
    int id;
    long long l, r;
};

void solve() {
    int n;
    cin >> n;
    vector<Segment> segments(n);
    vector<long long> l(n), r(n);
    long long initial_len_sum = 0;

    for (int i = 0; i < n; ++i) {
        segments[i].id = i;
        cin >> segments[i].l >> segments[i].r;
        l[i] = segments[i].l;
        r[i] = segments[i].r;
        initial_len_sum += r[i] - l[i];
    }

    if (n == 1) {
        cout << 0 << endl;
        return;
    }

    sort(segments.begin(), segments.end(), [](const Segment& a, const Segment& b) {
        if (a.l != b.l) return a.l < b.l;
        return a.r < b.r;
    });
    vector<Segment> min_l_segs = segments;

    sort(segments.begin(), segments.end(), [](const Segment& a, const Segment& b) {
        if (a.r != b.r) return a.r > b.r;
        return a.l > b.l;
    });
    vector<Segment> max_r_segs = segments;

    vector<bool> used(n, false);
    long long new_len_sum = 0;
    int l_ptr = 0, r_ptr = 0;
    int k = n / 2;

    for (int i = 0; i < k; ++i) {
        while (used[min_l_segs[l_ptr].id]) l_ptr++;
        while (used[max_r_segs[r_ptr].id]) r_ptr++;

        Segment sl = min_l_segs[l_ptr];
        Segment sr = max_r_segs[r_ptr];

        if (sl.id != sr.id) {
            new_len_sum += max(sl.r, sr.r) - min(sl.l, sr.l);
            used[sl.id] = true;
            used[sr.id] = true;
        } else {
            int l_ptr_next = l_ptr + 1;
            while (used[min_l_segs[l_ptr_next].id]) l_ptr_next++;
            Segment sl_next = min_l_segs[l_ptr_next];

            int r_ptr_next = r_ptr + 1;
            while (used[max_r_segs[r_ptr_next].id]) r_ptr_next++;
            Segment sr_next = max_r_segs[r_ptr_next];

            long long len1 = max(sl.r, sr_next.r) - min(sl.l, sr_next.l);
            long long len2 = max(sr.r, sl_next.r) - min(sr.l, sl_next.l);

            if (len1 >= len2) {
                new_len_sum += len1;
                used[sl.id] = true;
                used[sr_next.id] = true;
            } else {
                new_len_sum += len2;
                used[sr.id] = true;
                used[sl_next.id] = true;
            }
        }
    }

    cout << initial_len_sum + new_len_sum << endl;
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


