#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "kernel/metadata_json.hpp"
#include "remote/protocol.hpp"
#include "remote/sha256.hpp"
#include "remote/socket.hpp"

namespace remote = kora::detail::remote;

namespace {
std::string hex(const remote::Digest& digest) {
    std::string out;
    char buffer[3];
    for (const auto byte : digest) {
        std::snprintf(buffer, sizeof(buffer), "%02x", byte);
        out += buffer;
    }
    return out;
}

std::span<const std::uint8_t> text(std::string_view value) {
    return {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()};
}

struct Pair {
    remote::Socket client;
    remote::Socket server;
};

Pair connected_pair() {
    auto listener = remote::Socket::listen({"127.0.0.1", 0});
    EXPECT_TRUE(listener) << listener.error().describe();
    auto client =
        remote::Socket::connect({"127.0.0.1", listener.value().local_port()}, std::chrono::seconds(2));
    EXPECT_TRUE(client) << client.error().describe();
    auto server = listener.value().accept(std::chrono::seconds(2));
    EXPECT_TRUE(server && server.value().valid());
    return {std::move(client).value(), std::move(server).value()};
}
}  // namespace

TEST(RemoteProtocol, Sha256MatchesNistVectors) {
    EXPECT_EQ(hex(remote::sha256(text(""))),
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EXPECT_EQ(hex(remote::sha256(text("abc"))),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(hex(remote::sha256(text("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))),
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    const std::string million(1'000'000, 'a');
    EXPECT_EQ(hex(remote::sha256(text(million))),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(RemoteProtocol, HmacMatchesRfc4231) {
    EXPECT_EQ(hex(remote::hmac_sha256(text("Jefe"), text("what do ya want for nothing?"))),
              "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    const std::vector<std::uint8_t> long_key(131, 0xaa);
    EXPECT_EQ(
        hex(remote::hmac_sha256(long_key, text("Test Using Larger Than Block-Size Key - Hash Key First"))),
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
    auto digest = remote::sha256(text("x"));
    EXPECT_TRUE(remote::constant_time_equal(digest, digest));
    auto other = digest;
    other[31] ^= 1U;
    EXPECT_FALSE(remote::constant_time_equal(digest, other));
}

TEST(RemoteProtocol, Crc32MatchesStandardCheckValue) {
    const std::string check = "123456789";
    EXPECT_EQ(remote::crc32(std::as_bytes(std::span(check.data(), check.size()))), 0xCBF43926U);
}

TEST(RemoteProtocol, ParsesEndpointsAndRejectsMalformedOnes) {
    auto ipv4 = remote::parse_endpoint("192.168.1.5:7400");
    ASSERT_TRUE(ipv4);
    EXPECT_EQ(ipv4.value().host, "192.168.1.5");
    EXPECT_EQ(ipv4.value().port, 7400);
    auto ipv6 = remote::parse_endpoint("[::1]:80");
    ASSERT_TRUE(ipv6);
    EXPECT_EQ(ipv6.value().host, "::1");
    for (const auto* bad : {"host", ":7400", "host:", "host:70000", "host:12a", "[::1]80", "host:999999"})
        EXPECT_FALSE(remote::parse_endpoint(bad)) << bad;
}

TEST(RemoteProtocol, ReaderRejectsTruncatedPayloads) {
    remote::Writer writer;
    writer.u32(7);
    writer.string("kora");
    writer.u64(9);
    remote::Reader reader(writer.view());
    EXPECT_EQ(reader.u32().value(), 7U);
    EXPECT_EQ(reader.string().value(), "kora");
    EXPECT_EQ(reader.u64().value(), 9U);
    EXPECT_TRUE(reader.done());
    EXPECT_FALSE(reader.u8());

    // A length prefix larger than the remaining payload must fail, not over-read.
    remote::Writer lying;
    lying.u64(1ULL << 40U);
    remote::Reader truncated(lying.view());
    EXPECT_FALSE(truncated.bytes());
}

TEST(RemoteProtocol, FramesRoundTripAndCorruptionIsDetected) {
    auto pair = connected_pair();
    const std::vector<std::byte> payload{std::byte{1}, std::byte{2}, std::byte{3}};
    ASSERT_TRUE(
        remote::send_message(pair.client, remote::MessageType::Ping, 42, payload, std::chrono::seconds(2)));
    auto received = remote::receive_message(pair.server, 1024, std::chrono::seconds(2));
    ASSERT_TRUE(received) << received.error().describe();
    EXPECT_EQ(received.value().type, remote::MessageType::Ping);
    EXPECT_EQ(received.value().request_id, 42U);
    EXPECT_EQ(received.value().payload, payload);

    // Flip one payload byte after the header was written: the CRC must reject it.
    remote::Writer frame;
    frame.u32(remote::kMagic);
    frame.buffer().push_back(std::byte{remote::kProtocolVersion & 0xFFU});
    frame.buffer().push_back(std::byte{0});
    frame.buffer().push_back(std::byte{40});
    frame.buffer().push_back(std::byte{0});
    frame.u64(1);
    frame.u64(payload.size());
    frame.u32(remote::crc32(payload));
    frame.u32(0);
    auto corrupted = payload;
    corrupted[1] = std::byte{99};
    frame.buffer().insert(frame.buffer().end(), corrupted.begin(), corrupted.end());
    ASSERT_TRUE(pair.client.send_all(frame.view(), std::chrono::seconds(2)));
    EXPECT_FALSE(remote::receive_message(pair.server, 1024, std::chrono::seconds(2)));
}

TEST(RemoteProtocol, OversizedAndTruncatedFramesFailWithinTheDeadline) {
    {
        auto pair = connected_pair();
        const std::vector<std::byte> payload(64);
        ASSERT_TRUE(remote::send_message(pair.client, remote::MessageType::Ping, 1, payload,
                                         std::chrono::seconds(2)));
        EXPECT_FALSE(remote::receive_message(pair.server, 16, std::chrono::seconds(2)));
    }
    {
        auto pair = connected_pair();
        const std::array<std::byte, 10> partial{};
        ASSERT_TRUE(pair.client.send_all(partial, std::chrono::seconds(2)));
        const auto started = std::chrono::steady_clock::now();
        EXPECT_FALSE(remote::receive_message(pair.server, 1024, std::chrono::milliseconds(200)));
        EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(2));
        pair.client.close();
        EXPECT_FALSE(remote::receive_message(pair.server, 1024, std::chrono::seconds(2)));
    }
}

TEST(RemoteProtocol, KernelMetadataJsonRoundTrips) {
    kora::KernelMetadata metadata;
    metadata.kernel_id = "quote\"back\\slash\nnewline";
    metadata.local_size = {64, 2, 1};
    metadata.push_constant_size = 8;
    metadata.push_constants = {{"scale", "f32", 0, 4}, {"count", "u32", 4, 4}};
    metadata.bindings = {
        {0, 0, "in", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::Read},
        {0, 1, "out", kora::KernelBindingKind::StorageBuffer, kora::KernelAccess::ReadWrite}};
    metadata.specialization_constants = {
        {3, "flag", "bool", true}, {4, "neg", "i32", std::int64_t{-5}}, {5, "ratio", "f32", 0.25}};
    metadata.semantics = {"elementwise"};
    auto parsed = kora::parse_kernel_metadata(kora::detail::serialize_kernel_metadata(metadata));
    ASSERT_TRUE(parsed) << parsed.error().describe();
    EXPECT_EQ(parsed.value().kernel_id, metadata.kernel_id);
    EXPECT_EQ(parsed.value().local_size, metadata.local_size);
    ASSERT_EQ(parsed.value().bindings.size(), 2U);
    EXPECT_EQ(parsed.value().bindings[1].access, kora::KernelAccess::ReadWrite);
    ASSERT_EQ(parsed.value().push_constants.size(), 2U);
    EXPECT_EQ(parsed.value().push_constants[1].offset, 4U);
    ASSERT_EQ(parsed.value().specialization_constants.size(), 3U);
    EXPECT_EQ(std::get<bool>(parsed.value().specialization_constants[0].default_value), true);
    EXPECT_EQ(std::get<std::int64_t>(parsed.value().specialization_constants[1].default_value), -5);
    EXPECT_EQ(std::get<double>(parsed.value().specialization_constants[2].default_value), 0.25);
    EXPECT_EQ(parsed.value().semantics, metadata.semantics);
}
