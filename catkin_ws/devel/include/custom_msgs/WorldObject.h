// Generated by gencpp from file custom_msgs/WorldObject.msg
// DO NOT EDIT!


#ifndef CUSTOM_MSGS_MESSAGE_WORLDOBJECT_H
#define CUSTOM_MSGS_MESSAGE_WORLDOBJECT_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace custom_msgs
{
template <class ContainerAllocator>
struct WorldObject_
{
  typedef WorldObject_<ContainerAllocator> Type;

  WorldObject_()
    : objClass()
    , x(0.0)
    , y(0.0)
    , angle(0.0)
    , prob(0.0)  {
    }
  WorldObject_(const ContainerAllocator& _alloc)
    : objClass(_alloc)
    , x(0.0)
    , y(0.0)
    , angle(0.0)
    , prob(0.0)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _objClass_type;
  _objClass_type objClass;

   typedef float _x_type;
  _x_type x;

   typedef float _y_type;
  _y_type y;

   typedef float _angle_type;
  _angle_type angle;

   typedef double _prob_type;
  _prob_type prob;





  typedef boost::shared_ptr< ::custom_msgs::WorldObject_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::custom_msgs::WorldObject_<ContainerAllocator> const> ConstPtr;

}; // struct WorldObject_

typedef ::custom_msgs::WorldObject_<std::allocator<void> > WorldObject;

typedef boost::shared_ptr< ::custom_msgs::WorldObject > WorldObjectPtr;
typedef boost::shared_ptr< ::custom_msgs::WorldObject const> WorldObjectConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::custom_msgs::WorldObject_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::custom_msgs::WorldObject_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::custom_msgs::WorldObject_<ContainerAllocator1> & lhs, const ::custom_msgs::WorldObject_<ContainerAllocator2> & rhs)
{
  return lhs.objClass == rhs.objClass &&
    lhs.x == rhs.x &&
    lhs.y == rhs.y &&
    lhs.angle == rhs.angle &&
    lhs.prob == rhs.prob;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::custom_msgs::WorldObject_<ContainerAllocator1> & lhs, const ::custom_msgs::WorldObject_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace custom_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::custom_msgs::WorldObject_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::custom_msgs::WorldObject_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::custom_msgs::WorldObject_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::custom_msgs::WorldObject_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::custom_msgs::WorldObject_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::custom_msgs::WorldObject_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::custom_msgs::WorldObject_<ContainerAllocator> >
{
  static const char* value()
  {
    return "02bf617586369c005f6429b5354bd9ab";
  }

  static const char* value(const ::custom_msgs::WorldObject_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x02bf617586369c00ULL;
  static const uint64_t static_value2 = 0x5f6429b5354bd9abULL;
};

template<class ContainerAllocator>
struct DataType< ::custom_msgs::WorldObject_<ContainerAllocator> >
{
  static const char* value()
  {
    return "custom_msgs/WorldObject";
  }

  static const char* value(const ::custom_msgs::WorldObject_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::custom_msgs::WorldObject_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string objClass\n"
"float32 x\n"
"float32 y\n"
"float32 angle\n"
"float64 prob\n"
;
  }

  static const char* value(const ::custom_msgs::WorldObject_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::custom_msgs::WorldObject_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.objClass);
      stream.next(m.x);
      stream.next(m.y);
      stream.next(m.angle);
      stream.next(m.prob);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct WorldObject_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::custom_msgs::WorldObject_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::custom_msgs::WorldObject_<ContainerAllocator>& v)
  {
    s << indent << "objClass: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.objClass);
    s << indent << "x: ";
    Printer<float>::stream(s, indent + "  ", v.x);
    s << indent << "y: ";
    Printer<float>::stream(s, indent + "  ", v.y);
    s << indent << "angle: ";
    Printer<float>::stream(s, indent + "  ", v.angle);
    s << indent << "prob: ";
    Printer<double>::stream(s, indent + "  ", v.prob);
  }
};

} // namespace message_operations
} // namespace ros

#endif // CUSTOM_MSGS_MESSAGE_WORLDOBJECT_H
