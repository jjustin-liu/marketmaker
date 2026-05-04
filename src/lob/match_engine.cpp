// C++ limit order book with price-time priority matching.
//
// Mirrors src/lob/order_book.py. Same interface, same semantics, but
// uses std::map for ordered price levels and std::unordered_map for
// O(1) cancel lookup.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

enum class Side : int { BUY = 0, SELL = 1 };

struct Order {
    std::string order_id;
    Side side;
    double price;
    double size;
    int64_t timestamp;
};

struct Fill {
    std::string taker_order_id;
    std::string maker_order_id;
    double price;
    double size;
    int64_t timestamp;
};

using OrderPtr = std::shared_ptr<Order>;
using PriceLevel = std::deque<OrderPtr>;

class MatchEngine {
public:
    std::vector<Fill> insert(const std::string& order_id, Side side,
                             double price, double size, int64_t timestamp) {
        if (size <= 0.0) {
            throw std::invalid_argument("size must be positive");
        }
        if (order_map_.count(order_id)) {
            throw std::invalid_argument("duplicate order_id: " + order_id);
        }

        std::vector<Fill> fills;
        double remaining = match_(order_id, side, price, size, timestamp, fills);

        if (remaining > 0.0) {
            auto order = std::make_shared<Order>(
                Order{order_id, side, price, remaining, timestamp});
            rest_(order);
        }
        return fills;
    }

    bool cancel(const std::string& order_id) {
        auto it = order_map_.find(order_id);
        if (it == order_map_.end()) {
            return false;
        }
        Side side = it->second.first;
        double price = it->second.second;
        order_map_.erase(it);

        if (side == Side::BUY) {
            erase_from_level_(bids_, price, order_id);
        } else {
            erase_from_level_(asks_, price, order_id);
        }
        return true;
    }

    std::optional<double> best_bid() const {
        if (bids_.empty()) return std::nullopt;
        return bids_.begin()->first;
    }

    std::optional<double> best_ask() const {
        if (asks_.empty()) return std::nullopt;
        return asks_.begin()->first;
    }

    std::vector<std::pair<double, double>> depth(Side side, int levels) const {
        std::vector<std::pair<double, double>> out;
        if (side == Side::BUY) {
            collect_depth_(bids_, levels, out);
        } else {
            collect_depth_(asks_, levels, out);
        }
        return out;
    }

    void apply_diff(Side side, double price, double qty) {
        if (side == Side::BUY) {
            apply_diff_impl_(bids_, side, price, qty);
        } else {
            apply_diff_impl_(asks_, side, price, qty);
        }
    }

private:
    // bids: highest price first. asks: lowest price first.
    std::map<double, PriceLevel, std::greater<double>> bids_;
    std::map<double, PriceLevel, std::less<double>> asks_;
    std::unordered_map<std::string, std::pair<Side, double>> order_map_;

    template <typename Book>
    double match_one_side_(Book& opposite, const std::string& taker_id,
                           Side side, double price, double remaining,
                           int64_t timestamp, std::vector<Fill>& fills) {
        while (remaining > 0.0 && !opposite.empty()) {
            auto best_it = opposite.begin();
            double best_price = best_it->first;

            bool crosses = (side == Side::BUY) ? (best_price <= price)
                                               : (best_price >= price);
            if (!crosses) break;

            PriceLevel& queue = best_it->second;
            while (remaining > 0.0 && !queue.empty()) {
                OrderPtr maker = queue.front();
                double trade_size = std::min(remaining, maker->size);

                fills.push_back(Fill{taker_id, maker->order_id, best_price,
                                     trade_size, timestamp});
                maker->size -= trade_size;
                remaining -= trade_size;

                if (maker->size == 0.0) {
                    order_map_.erase(maker->order_id);
                    queue.pop_front();
                }
            }
            if (queue.empty()) {
                opposite.erase(best_it);
            }
        }
        return remaining;
    }

    double match_(const std::string& taker_id, Side side, double price,
                  double size, int64_t timestamp, std::vector<Fill>& fills) {
        if (side == Side::BUY) {
            return match_one_side_(asks_, taker_id, side, price, size,
                                   timestamp, fills);
        } else {
            return match_one_side_(bids_, taker_id, side, price, size,
                                   timestamp, fills);
        }
    }

    void rest_(const OrderPtr& order) {
        if (order->side == Side::BUY) {
            bids_[order->price].push_back(order);
        } else {
            asks_[order->price].push_back(order);
        }
        order_map_[order->order_id] = {order->side, order->price};
    }

    template <typename Book>
    void erase_from_level_(Book& book, double price,
                           const std::string& order_id) {
        auto it = book.find(price);
        if (it == book.end()) return;
        PriceLevel& queue = it->second;
        for (auto qit = queue.begin(); qit != queue.end(); ++qit) {
            if ((*qit)->order_id == order_id) {
                queue.erase(qit);
                break;
            }
        }
        if (queue.empty()) {
            book.erase(it);
        }
    }

    template <typename Book>
    void collect_depth_(const Book& book, int levels,
                        std::vector<std::pair<double, double>>& out) const {
        int taken = 0;
        for (const auto& [price, queue] : book) {
            if (taken >= levels) break;
            double total = 0.0;
            for (const auto& o : queue) total += o->size;
            out.emplace_back(price, total);
            ++taken;
        }
    }

    template <typename Book>
    void apply_diff_impl_(Book& book, Side side, double price, double qty) {
        if (qty == 0.0) {
            book.erase(price);
            return;
        }
        std::string synthetic_id = "_diff_" +
            std::string(side == Side::BUY ? "BUY" : "SELL") + "_" +
            std::to_string(price);
        auto synthetic = std::make_shared<Order>(
            Order{synthetic_id, side, price, qty, 0});
        PriceLevel q;
        q.push_back(synthetic);
        book[price] = std::move(q);
    }
};

PYBIND11_MODULE(_match_engine, m) {
    m.doc() = "C++ limit order book with price-time priority";

    py::enum_<Side>(m, "Side")
        .value("BUY", Side::BUY)
        .value("SELL", Side::SELL)
        .export_values();

    py::class_<Fill>(m, "Fill")
        .def_readonly("taker_order_id", &Fill::taker_order_id)
        .def_readonly("maker_order_id", &Fill::maker_order_id)
        .def_readonly("price", &Fill::price)
        .def_readonly("size", &Fill::size)
        .def_readonly("timestamp", &Fill::timestamp)
        .def("__repr__", [](const Fill& f) {
            return "Fill(taker=" + f.taker_order_id +
                   ", maker=" + f.maker_order_id +
                   ", price=" + std::to_string(f.price) +
                   ", size=" + std::to_string(f.size) + ")";
        });

    py::class_<MatchEngine>(m, "MatchEngine")
        .def(py::init<>())
        .def("insert", &MatchEngine::insert,
             py::arg("order_id"), py::arg("side"), py::arg("price"),
             py::arg("size"), py::arg("timestamp"))
        .def("cancel", &MatchEngine::cancel, py::arg("order_id"))
        .def("best_bid", &MatchEngine::best_bid)
        .def("best_ask", &MatchEngine::best_ask)
        .def("depth", &MatchEngine::depth, py::arg("side"), py::arg("levels"))
        .def("apply_diff", &MatchEngine::apply_diff,
             py::arg("side"), py::arg("price"), py::arg("qty"));
}
